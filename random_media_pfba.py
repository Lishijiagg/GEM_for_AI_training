#!/usr/bin/env python3
"""
random_media_pfba.py
Simulate one GEM (e.g. an AGORA/AGORA2/APOLLO model) under random nutrient
conditions (presence/absence design).

Every exchange of the model except the fixed compounds (h2o, h) and the
excluded ones (o2) is a candidate nutrient: carbon sources, amino acids,
vitamins, ions, NH4, phosphate, sulfate, drugs. Nothing is supplied by default,
so auxotrophies and missing inorganic nutrients show up as no-growth conditions.

Steps
  0. Problem A (model defect): if biomass cannot be made with every candidate
     open, add sinks for biomass precursors without a synthesis route
     (see gemtools.repair_biomass). Models that cannot be repaired are skipped.
  1. Close all uptakes; open only h2o and h.
  2. For each condition, choose which candidates are present (lower bound -1000)
     with one of two designs:
       subset  : presence probability r ~ U(0, 1), each candidate present with
                 probability r (sparse to rich media)
       dropout : start from all candidates and remove m ~ U(1, m_max) of them
                 (rich media with a few missing nutrients; informative for
                 auxotrophies)
     --frac-dropout sets the share of dropout conditions (default 0.5).
  3. FBA (biomass) and pFBA (fluxes); one output folder per condition.

Usage
  python3 random_media_pfba.py --model model.xml --out results/strain --n 1000

Output
  run_info.txt, simulation.log, candidate_substrates.tsv, conditions_summary.tsv,
  substrates_presence.tsv, products_presence.tsv and cond_XXXX/ folders with
  result.tsv, medium.tsv, products.tsv, exchanges.tsv.
Fluxes in mmol gDW-1 h-1; negative = uptake, positive = secretion.
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from cobra.flux_analysis import pfba

import gemtools as gt

logging.getLogger("cobra").setLevel(logging.ERROR)
NOT_PRODUCTS = {"h2o", "h"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="SBML file")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--n", type=int, default=1000, help="number of conditions")
    ap.add_argument("--frac-dropout", type=float, default=0.5,
                    help="share of conditions made with the dropout design (0-1)")
    ap.add_argument("--m-max", type=int, default=20,
                    help="dropout design: max number of candidates removed")
    ap.add_argument("--lb", type=float, default=1000,
                    help="uptake bound of a present nutrient (lower bound = -lb)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fixed", default="h2o:1000,h:1000",
                    help="always-open compounds, metabolite ID without compartment : bound")
    ap.add_argument("--exclude", default="o2",
                    help="compounds never supplied, comma separated")
    a = ap.parse_args()

    fixed = gt.parse_pairs(a.fixed)
    exclude = set(filter(None, (x.strip() for x in a.exclude.split(","))))
    model = gt.load(a.model)
    biomass = gt.biomass_id(model)
    table, candidates = gt.setup_exchanges(model, fixed, exclude)

    status, sinks = gt.repair_biomass(model, candidates, a.lb)
    print(f"model check: {status}; sinks added: {sinks or 'none'}")
    os.makedirs(a.out, exist_ok=True)
    if status == "unrepairable":
        with open(os.path.join(a.out, "run_info.txt"), "w") as f:
            f.write(f"model\t{os.path.abspath(a.model)}\nstatus\tunrepairable: no biomass "
                    f"even with all candidates open and sinks for all precursors\n")
        sys.exit("model cannot produce biomass; skipped")

    ex = pd.DataFrame(table, columns=["exchange_id", "metabolite", "name"])
    ex["role"] = np.where(ex.metabolite.isin(fixed), "fixed",
                          np.where(ex.exchange_id.isin(candidates), "candidate", "excluded"))
    with open(os.path.join(a.out, "run_info.txt"), "w") as f:
        f.write(f"model\t{os.path.abspath(a.model)}\nbiomass_reaction\t{biomass}\n"
                f"model_check\t{status}\nadded_precursor_sinks\t{sinks or 'none'}\n"
                f"n_conditions\t{a.n}\nfrac_dropout\t{a.frac_dropout}\nm_max\t{a.m_max}\n"
                f"nutrient_lower_bound\t{-a.lb}\nseed\t{a.seed}\nfixed\t{fixed}\n"
                f"excluded\t{sorted(exclude)}\nn_candidates\t{len(candidates)}\n")
    ex[ex.role == "candidate"].to_csv(os.path.join(a.out, "candidate_substrates.tsv"),
                                      sep="\t", index=False)

    rng = np.random.default_rng(a.seed)
    nc = len(candidates)
    n_drop = int(round(a.n * a.frac_dropout))
    designs = ["dropout"] * n_drop + ["subset"] * (a.n - n_drop)
    rng.shuffle(designs)

    summary, presence, secreted = [], {}, {}
    log = open(os.path.join(a.out, "simulation.log"), "w")
    log.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} start: model {os.path.basename(a.model)}, "
              f"{a.n} conditions, {nc} candidates, sinks {sinks or 'none'}\n")
    log.flush()
    t_run = time.perf_counter()

    for c, design in enumerate(designs, start=1):
        cid = f"cond_{c:04d}"
        t0 = time.perf_counter()
        if design == "subset":
            r = rng.uniform(0, 1)
            present = rng.random(nc) < r
        else:
            m = int(rng.integers(1, min(a.m_max, nc) + 1))
            present = np.ones(nc, bool)
            present[rng.choice(nc, m, replace=False)] = False
        chosen = [rid for rid, p in zip(candidates, present) if p]

        with model:
            for rid in chosen:
                model.reactions.get_by_id(rid).lower_bound = -a.lb
            t1 = time.perf_counter()
            sol = model.optimize()
            t2 = time.perf_counter()
            growth = sol.objective_value if sol.status == "optimal" else np.nan
            if sol.status == "optimal" and growth > gt.TOL:
                flux = pfba(model).fluxes
            else:
                flux = pd.Series(0.0, index=[r.id for r in model.reactions])
            t3 = time.perf_counter()
            bounds = {r.id: r.bounds for r in model.exchanges}

        d = os.path.join(a.out, cid)
        os.makedirs(d, exist_ok=True)
        t = ex.copy()
        t.loc[t.exchange_id.isin(chosen), "role"] = "substrate_set"
        t["lower_bound"] = [bounds[x][0] for x in t.exchange_id]
        t["upper_bound"] = [bounds[x][1] for x in t.exchange_id]
        t["flux"] = flux.reindex(t.exchange_id).values.round(6)
        t.loc[t.flux.abs() < gt.TOL, "flux"] = 0.0
        t.to_csv(os.path.join(d, "exchanges.tsv"), sep="\t", index=False)
        t[t.role == "substrate_set"][["exchange_id", "metabolite", "name", "lower_bound", "flux"]] \
            .to_csv(os.path.join(d, "medium.tsv"), sep="\t", index=False)
        t[t.flux > 0].sort_values("flux", ascending=False)[["exchange_id", "metabolite", "name", "flux"]] \
            .to_csv(os.path.join(d, "products.tsv"), sep="\t", index=False)
        res = dict(condition=cid, design=design, status=sol.status, biomass_reaction=biomass,
                   biomass=round(float(growth), 6) if np.isfinite(growth) else np.nan,
                   n_substrates_set=len(chosen),
                   n_substrates_used=int((t[t.role == "substrate_set"].flux < 0).sum()),
                   n_products=int((t.flux > 0).sum()),
                   fba_s=round(t2 - t1, 4), pfba_s=round(t3 - t2, 4))
        t4 = time.perf_counter()
        res["write_s"] = round(t4 - t3, 4)
        res["total_s"] = round(t4 - t0, 4)
        pd.DataFrame([res]).to_csv(os.path.join(d, "result.tsv"), sep="\t", index=False)
        summary.append(res)
        presence[cid] = set(t.loc[t.role == "substrate_set", "metabolite"])
        secreted[cid] = set(t.loc[(t.flux > 0) & ~t.metabolite.isin(NOT_PRODUCTS), "metabolite"])
        log.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {cid} design={design} status={sol.status} "
                  f"biomass={res['biomass']} substrates={len(chosen)} fba={res['fba_s']}s "
                  f"pfba={res['pfba_s']}s write={res['write_s']}s total={res['total_s']}s\n")
        log.flush()
        if c % 100 == 0:
            print(f"{c}/{a.n} done")

    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(a.out, "conditions_summary.tsv"), sep="\t", index=False)
    wall = time.perf_counter() - t_run
    log.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} finished: {len(s)} conditions in {wall:.1f}s; "
              f"mean per condition {s.total_s.mean():.3f}s (fba {s.fba_s.mean():.3f}s, "
              f"pfba {s.pfba_s.mean():.3f}s, write {s.write_s.mean():.3f}s)\n")
    log.close()

    grow = (s.set_index("condition").biomass > gt.TOL).astype(int).rename("biomass_produced")
    subs = sorted(ex.loc[ex.role == "candidate", "metabolite"])
    S = pd.DataFrame([[int(x in presence[c]) for x in subs] for c in s.condition],
                     index=s.condition, columns=subs)
    prods = sorted(set().union(*secreted.values()))
    P = pd.DataFrame([[int(x in secreted[c]) for x in prods] for c in s.condition],
                     index=s.condition, columns=prods)
    pd.concat([grow, S], axis=1).to_csv(os.path.join(a.out, "substrates_presence.tsv"), sep="\t")
    pd.concat([grow, P], axis=1).to_csv(os.path.join(a.out, "products_presence.tsv"), sep="\t")
    by = s.assign(g=s.biomass > gt.TOL).groupby("design").g.mean()
    print(f"biomass produced in {grow.mean():.1%} of {len(s)} conditions "
          f"({', '.join(f'{k} {v:.1%}' for k, v in by.items())})")


if __name__ == "__main__":
    main()
