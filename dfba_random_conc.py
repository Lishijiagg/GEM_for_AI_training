#!/usr/bin/env python3
"""
dfba_random_conc.py
Stage 2: dynamic FBA (dFBA) of one GEM in random virtual media.

Input: the stage-1 result folder of the same model (random_media_pfba.py).
Only substrate combinations that produced biomass in stage 1 are used. For each
of them, --n-conc random concentration sets are drawn: every nutrient present in
that combination gets an initial concentration ~ log-uniform(conc_min, conc_max)
mM; absent nutrients start at 0.

Batch culture, 1 L, no per-compound uptake kinetics (no Vmax, no Km); one
global limit on total carbon uptake (--carbon-cap, default 60 mmol C gDW-1 h-1;
0 switches it off). The ODEs are integrated every --dt (default 0.01 h); the
state is written every --record-dt (default 0.1 h). At every time step:
  1. uptake limit from the medium:
       lower bound of EX_i = -C_i / (X * dt)        [mmol gDW-1 h-1]
     (C_i in mM = mmol L-1, X in gDW L-1, dt in h): the cells may take up at
     most what is in the medium during this step. Compounds with C_i = 0 cannot
     be taken up. Secreted products enter the medium and can be taken up later.
     h2o and h stay open (-1000), o2 stays closed.
  2. FBA (max biomass, subject to the carbon cap) gives the growth rate mu;
     fluxes from FBA or pFBA.
  3. explicit Euler update (ODEs dX/dt = mu X, dC_i/dt = v_i X):
       C_i(t+dt) = C_i(t) + v_i X(t) dt
       X(t+dt)   = X(t) + mu X(t) dt
     With the uptake limit above, C_i never becomes negative.
  4. If mu = 0, nothing changes any more (no maintenance reaction is forced),
     the simulation stops and the last state is kept until t_end.
Both FBA and pFBA are run for every concentration set (--methods).

Model repair (Problem A) is identical to stage 1 (gemtools.repair_biomass).

Usage
  python3 dfba_random_conc.py --model models/X.xml --stage1 results/X \
      --out results_dfba/X --processes 16

Output (results_dfba/<model>/)
  run_info.txt                 parameters
  dfba.log                     one line per run with timing
  dfba_summary.tsv             one row per run x method
  dfba_initial_conc.tsv        run x nutrient, initial concentration (mM)
  dfba_net_change.tsv          run x method x compound, final - initial concentration (mM)
  With --n-chunks N --chunk i, only every N-th concentration set is run and the
  five files above get the suffix _chunkIIIofNNN; merge_dfba.py combines them.
  cond_XXXX/conc_YY/
      initial_medium.tsv       nutrients with initial concentration (mM)
      fba/  and  pfba/
          timecourse.tsv       time, biomass (gDW L-1), growth rate (h-1), every record_dt
          concentrations.tsv   time x compound (mM), compounds that are ever non-zero
          fluxes.tsv           time x exchange (mmol gDW-1 h-1), exchanges that are ever non-zero
          summary.tsv          final biomass, max growth rate, stop time, consumed / produced compounds
"""
import argparse
import logging
import os
import time
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
from optlang.symbolics import Zero

import gemtools as gt

logging.getLogger("cobra").setLevel(logging.ERROR)
TOL = gt.TOL
CONC_TOL = 1e-9     # mM; smaller concentrations are set to 0

_W = {}


def _init(cfg):
    gt.set_solver(cfg["solver"])          # also in worker processes
    model = gt.load(cfg["model"])
    table, candidates = gt.setup_exchanges(model, cfg["fixed"], cfg["exclude"])
    status, sinks = gt.repair_biomass(model, candidates)
    met_of = {rid: b for rid, b, _ in table}
    if cfg.get("carbon_cap", 0) > 0:
        # optional: total carbon uptake <= cap (mmol C gDW-1 h-1)
        terms = [next(iter(model.reactions.get_by_id(r).metabolites)).elements.get("C", 0)
                 * model.reactions.get_by_id(r).reverse_variable for r in candidates]
        model.add_cons_vars(model.problem.Constraint(sum(terms), ub=cfg["carbon_cap"],
                                                     name="carbon_uptake_cap"))
    _W.update(model=model, cand=candidates, met_of=met_of,
              rxns=[model.reactions.get_by_id(r) for r in candidates], cfg=cfg,
              status=status, sinks=sinks)


def simulate(c0, method):
    """c0: initial concentrations aligned with candidates (mM).
    The ODEs are integrated every dt; the state is recorded every record_dt.
    Returns times, biomass, growth rates, concentration and flux matrices at the
    recorded time points, stop time, max growth rate and its time (all steps)."""
    cfg = _W["cfg"]
    # fresh copy of the model for every run: the LP solver keeps its last basis,
    # and with alternative optima the FBA path would otherwise depend on which
    # run the same worker solved before (results would depend on --processes)
    m = gt.single_thread(_W["model"].copy())
    rxns = [m.reactions.get_by_id(r) for r in _W["cand"]]
    if method == "pfba":
        # second copy holding the pFBA problem, built once per run: growth fixed
        # to the FBA optimum by a constraint whose lower bound is updated each
        # step, objective = minimal total flux (as cobra.flux_analysis.pfba,
        # which rebuilds this problem at every call and is about 8x slower)
        mp = gt.single_thread(_W["model"].copy())
        rxns_p = [mp.reactions.get_by_id(r) for r in _W["cand"]]
        fix = mp.problem.Constraint(mp.solver.objective.expression, lb=0.0,
                                    name="fixed_growth")
        mp.add_cons_vars(fix)
        mp.objective = mp.problem.Objective(Zero, direction="min", sloppy=True)
        mp.objective.set_linear_coefficients(
            {v: 1.0 for r in mp.reactions for v in (r.forward_variable, r.reverse_variable)})
    dt, nsteps = cfg["dt"], int(round(cfg["t_end"] / cfg["dt"]))
    rec = max(1, int(round(cfg["record_dt"] / dt)))
    C = c0.astype(float).copy()
    X = cfg["x0"]
    times, Xs, mus, Cs, Vs = [], [], [], [], []
    stop, mu_max, t_mu_max = cfg["t_end"], 0.0, np.nan

    def record(t, X, mu, C, v):
        times.append(t); Xs.append(X); mus.append(mu)
        Cs.append(C.copy()); Vs.append(v)

    for k in range(nsteps + 1):
        t = round(k * dt, 10)
        if k == nsteps:                       # last time point: state only
            record(t, X, np.nan, C, np.zeros(len(rxns)))
            break
        lbs = np.where(C > CONC_TOL, -C / (X * dt), 0.0)
        for r, lb in zip(rxns, lbs):
            r.lower_bound = lb
        mu = m.slim_optimize(error_value=0.0)
        mu = mu if mu > TOL else 0.0
        if mu > 0:
            v = None
            if method == "pfba":
                for r, lb in zip(rxns_p, lbs):
                    r.lower_bound = lb
                fix.lb = mu
                if mp.slim_optimize(error_value=None) is not None:
                    v = np.array([r.flux for r in rxns_p])
            if v is None:                     # FBA, or pFBA step not solved
                v = np.array([r.flux for r in rxns])
            v[np.abs(v) < TOL] = 0.0
            v = np.maximum(v, lbs)            # guard against solver tolerance
        else:
            v = np.zeros(len(rxns))
        if mu > mu_max:
            mu_max, t_mu_max = mu, t
        if k % rec == 0:
            record(t, X, mu, C, v)
        if mu == 0:
            stop = t
            # state is constant from here on: pad recorded time points to t_end
            for kk in range(k + 1, nsteps + 1):
                if kk % rec == 0 or kk == nsteps:
                    record(round(kk * dt, 10), X, 0.0 if kk < nsteps else np.nan, C,
                           np.zeros(len(rxns)))
            break
        C = C + v * X * dt
        C[C < CONC_TOL] = 0.0
        X = X + mu * X * dt
    for r in rxns:
        r.lower_bound = 0.0
    return (np.array(times), np.array(Xs), np.array(mus),
            np.vstack(Cs), np.vstack(Vs), stop, mu_max, t_mu_max)


def run_one(spec):
    """spec: (cond, conc_id, c0 vector, output folder)"""
    cond, conc_id, c0, d = spec
    cand, met_of, cfg = _W["cand"], _W["met_of"], _W["cfg"]
    mets = [met_of[r] for r in cand]
    os.makedirs(d, exist_ok=True)
    init = pd.DataFrame({"exchange_id": cand, "metabolite": mets, "initial_conc_mM": c0})
    init[init.initial_conc_mM > 0].to_csv(os.path.join(d, "initial_medium.tsv"),
                                         sep="\t", index=False)
    rows, changes = [], []
    for method in cfg["methods"]:
        t0 = time.perf_counter()
        T, X, MU, Cm, Vm, stop, mu_max, t_mu_max = simulate(c0, method)
        t1 = time.perf_counter()
        md = os.path.join(d, method)
        os.makedirs(md, exist_ok=True)
        pd.DataFrame({"time_h": T, "biomass_gDW_L": X, "growth_rate_h": MU}) \
            .to_csv(os.path.join(md, "timecourse.tsv"), sep="\t", index=False)
        keepc = np.where((Cm > 0).any(axis=0))[0]
        cdf = pd.DataFrame(Cm[:, keepc], columns=[mets[i] for i in keepc])
        cdf.insert(0, "time_h", T)
        cdf.to_csv(os.path.join(md, "concentrations.tsv"), sep="\t", index=False,
                   float_format="%.6g")
        keepv = np.where((Vm != 0).any(axis=0))[0]
        vdf = pd.DataFrame(Vm[:, keepv], columns=[cand[i] for i in keepv])
        vdf.insert(0, "time_h", T)
        vdf.to_csv(os.path.join(md, "fluxes.tsv"), sep="\t", index=False, float_format="%.6g")
        net = Cm[-1] - Cm[0]
        consumed = [mets[i] for i in np.where(net < -CONC_TOL)[0]]
        produced = [mets[i] for i in np.where(net > CONC_TOL)[0]]
        res = dict(condition=cond, conc_set=conc_id, method=method,
                   initial_biomass=X[0], final_biomass=X[-1],
                   max_growth_rate=mu_max, time_max_growth=t_mu_max,
                   stop_time=stop, n_nutrients=int((c0 > 0).sum()),
                   n_consumed=len(consumed), n_produced=len(produced),
                   produced=";".join(produced),
                   sim_s=round(t1 - t0, 3))
        t2 = time.perf_counter()
        res["write_s"] = round(t2 - t1, 3)
        pd.DataFrame([res]).to_csv(os.path.join(md, "summary.tsv"), sep="\t", index=False)
        rows.append(res)
        changes.append(dict(condition=cond, conc_set=conc_id, method=method,
                            **{mets[i]: net[i] for i in np.where(np.abs(net) > CONC_TOL)[0]}))
    return rows, changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="SBML file (same as stage 1)")
    ap.add_argument("--stage1", required=True, help="stage-1 result folder of this model")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--n-conc", type=int, default=10, help="concentration sets per combination")
    ap.add_argument("--conc-min", type=float, default=0.01, help="min initial concentration (mM)")
    ap.add_argument("--conc-max", type=float, default=10.0, help="max initial concentration (mM)")
    ap.add_argument("--x0", type=float, default=0.01, help="initial biomass (gDW L-1)")
    ap.add_argument("--t-end", type=float, default=24.0, help="simulated time (h)")
    ap.add_argument("--dt", type=float, default=0.01, help="integration time step (h)")
    ap.add_argument("--record-dt", type=float, default=0.1,
                    help="interval of the time points written to the tables (h)")
    ap.add_argument("--methods", default="fba,pfba", help="fba, pfba or fba,pfba")
    ap.add_argument("--carbon-cap", type=float, default=60,
                    help="max total carbon uptake, mmol C gDW-1 h-1 (0 = off)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--processes", type=int, default=1)
    ap.add_argument("--max-conditions", type=int, default=0,
                    help="use only the first N growing combinations (0 = all; for tests)")
    ap.add_argument("--n-chunks", type=int, default=1,
                    help="split the concentration sets of this model into N parts (array tasks)")
    ap.add_argument("--chunk", type=int, default=1,
                    help="which part to run, 1..n-chunks; merge parts with merge_dfba.py")
    ap.add_argument("--fixed", default="h2o:1000,h:1000")
    ap.add_argument("--exclude", default="o2")
    ap.add_argument("--solver", default="auto",
                    help="LP solver: auto (gurobi > cplex > glpk, first installed), cplex, glpk, gurobi")
    a = ap.parse_args()
    if not 1 <= a.chunk <= a.n_chunks:
        raise SystemExit("--chunk must be between 1 and --n-chunks")
    # file-name suffix of this part (empty when the model runs as one part)
    sfx = "" if a.n_chunks == 1 else f"_chunk{a.chunk:03d}of{a.n_chunks:03d}"

    fixed = gt.parse_pairs(a.fixed)
    exclude = set(filter(None, (x.strip() for x in a.exclude.split(","))))
    methods = [x.strip() for x in a.methods.split(",") if x.strip()]
    cfg = dict(model=a.model, fixed=fixed, exclude=exclude, dt=a.dt, t_end=a.t_end,
               x0=a.x0, methods=methods, carbon_cap=a.carbon_cap,
               record_dt=a.record_dt, solver=a.solver)

    # growing combinations from stage 1
    S = pd.read_csv(os.path.join(a.stage1, "substrates_presence.tsv"), sep="\t", index_col=0)
    grow = S[S.biomass_produced == 1].drop(columns="biomass_produced")
    if a.max_conditions:
        grow = grow.iloc[:a.max_conditions]

    _init(cfg)
    cand, met_of = _W["cand"], _W["met_of"]
    mets = [met_of[r] for r in cand]
    missing = set(grow.columns) - set(mets)
    if missing:
        raise SystemExit(f"stage-1 nutrients not found in the model: {sorted(missing)[:5]} ...")
    print(f"solver: {gt.solver_name(_W['model'].solver.interface)}; "
          f"model check: {_W['status']}; sinks: {_W['sinks'] or 'none'}; "
          f"{len(grow)} growing combinations x {a.n_conc} concentration sets x {methods}")

    rng = np.random.default_rng(a.seed)
    lo, hi = np.log10(a.conc_min), np.log10(a.conc_max)
    specs, init_rows = [], []
    for cond, row in grow.iterrows():
        present = np.array([row.get(mt, 0) == 1 for mt in mets])
        for j in range(1, a.n_conc + 1):
            c0 = np.where(present, 10 ** rng.uniform(lo, hi, len(cand)), 0.0)
            cid = f"conc_{j:02d}"
            specs.append((cond, cid, c0, os.path.join(a.out, cond, cid)))
            init_rows.append(dict(condition=cond, conc_set=cid,
                                  **{mets[i]: c0[i] for i in np.where(present)[0]}))
    # all concentration sets are drawn first (same random numbers for any
    # number of parts); then this part takes every n-th set
    n_all = len(specs)
    keep = [j for j in range(n_all) if j % a.n_chunks == a.chunk - 1]
    specs = [specs[j] for j in keep]
    init_rows = [init_rows[j] for j in keep]
    if a.n_chunks > 1:
        print(f"part {a.chunk} of {a.n_chunks}: {len(specs)} of {n_all} concentration sets")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"run_info{sfx}.txt"), "w") as f:
        f.write(f"model\t{os.path.abspath(a.model)}\nstage1\t{os.path.abspath(a.stage1)}\n"
                f"solver\t{gt.solver_name(_W['model'].solver.interface)}\n"
                f"model_check\t{_W['status']}\nadded_precursor_sinks\t{_W['sinks'] or 'none'}\n"
                f"growing_combinations\t{len(grow)}\nconc_sets_per_combination\t{a.n_conc}\n"
                f"initial_conc\tlog-uniform {a.conc_min}-{a.conc_max} mM\n"
                f"initial_biomass\t{a.x0} gDW/L\nt_end\t{a.t_end} h\ndt\t{a.dt} h\n"
                f"record_dt\t{a.record_dt} h\n"
                f"methods\t{methods}\nuptake_limit\tlb = -C/(X*dt), no Vmax/Km\n"
                f"carbon_cap\t{a.carbon_cap or 'off'}\n"
                f"fixed\t{fixed}\nexcluded\t{sorted(exclude)}\nseed\t{a.seed}\n"
                f"total_conc_sets\t{n_all}\nn_chunks\t{a.n_chunks}\nchunk\t{a.chunk}\n"
                f"conc_sets_in_chunk\t{len(specs)}\n")
    pd.DataFrame(init_rows).fillna(0).to_csv(os.path.join(a.out, f"dfba_initial_conc{sfx}.tsv"),
                                             sep="\t", index=False, float_format="%.6g")

    log = open(os.path.join(a.out, f"dfba{sfx}.log"), "w")
    log.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} start: {len(specs)} concentration sets, "
              f"methods {methods}, processes {a.processes}\n")
    log.flush()
    t_run = time.perf_counter()
    rows, changes = [], []
    if a.processes > 1:
        pool = Pool(a.processes, initializer=_init, initargs=(cfg,))
        # one concentration set per task: each set takes seconds to minutes,
        # so idle workers pick up the next set as soon as they finish
        it = pool.imap_unordered(run_one, specs, chunksize=1)
    else:
        pool, it = None, map(run_one, specs)
    for i, (r, ch) in enumerate(it, start=1):
        rows += r
        changes += ch
        for x in r:
            log.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {x['condition']}/{x['conc_set']} "
                      f"{x['method']} final_biomass={x['final_biomass']:.4g} "
                      f"max_mu={x['max_growth_rate']:.4g} stop={x['stop_time']}h "
                      f"sim={x['sim_s']}s write={x['write_s']}s\n")
        log.flush()
        if i % 50 == 0:
            print(f"{i}/{len(specs)} concentration sets done")
    if pool:
        pool.close(); pool.join()

    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(a.out, f"dfba_summary{sfx}.tsv"), sep="\t", index=False)
    pd.DataFrame(changes).fillna(0).to_csv(os.path.join(a.out, f"dfba_net_change{sfx}.tsv"),
                                           sep="\t", index=False, float_format="%.6g")
    wall = time.perf_counter() - t_run
    log.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} finished: {len(specs)} sets in {wall:.1f}s\n")
    log.close()
    for mth, g in s.groupby("method"):
        print(f"{mth}: final biomass median {g.final_biomass.median():.4g} gDW/L, "
              f"max growth rate median {g.max_growth_rate.median():.4g} h-1, "
              f"stop time median {g.stop_time.median():.3g} h")
    print(f"done in {wall:.0f} s")


if __name__ == "__main__":
    main()
