"""
gemtools.py
Shared helpers for random_media_pfba.py and check_model.py.
"""
import re

import cobra

TOL = 1e-6
FIXED_DEFAULT = {"h2o": 1000.0, "h": 1000.0}   # always open (solvent and protons)
EXCLUDE_DEFAULT = {"o2"}                          # never supplied (anaerobic)


def base_id(met):
    """Metabolite ID without the extracellular compartment tag."""
    return re.sub(r"(\[e0?\]|\(e0?\)|_e0?)$", "", met.id)


def biomass_id(model):
    return [r.id for r in model.reactions if r.objective_coefficient != 0][0]


def setup_exchanges(model, fixed, exclude):
    """Close every uptake, block biomass re-import, open the fixed compounds.
    Returns a list of (exchange_id, metabolite_base_id, name) for every exchange
    and the list of candidate exchange IDs that can be sampled."""
    for r in model.exchanges:
        r.lower_bound = 0.0
    for r in model.boundary:
        if "biomass" in r.id.lower():
            r.lower_bound = 0.0
    table, candidates = [], []
    for r in model.exchanges:
        met = next(iter(r.metabolites))
        b = base_id(met)
        table.append((r.id, b, met.name))
        if b in fixed:
            r.lower_bound = -fixed[b]
        elif b not in exclude:
            candidates.append(r.id)
    return table, candidates


def repair_biomass(model, candidates, lb_open=1000.0):
    """Problem A: if the model cannot make biomass with every candidate
    substrate open, add sinks (lb -1) for biomass precursors that cannot be
    synthesised.
    Step 1: precursors used by no reaction except the biomass reaction.
    Step 2 (only if step 1 is not enough): sinks for all precursors, then
    remove them one by one and keep only those needed (step-1 sinks are tested
    last so they are kept preferentially).
    Returns (status, sinks) with status in {"ok", "repaired", "unrepairable"}."""
    bio = model.reactions.get_by_id(biomass_id(model))
    with model:
        for rid in candidates:
            model.reactions.get_by_id(rid).lower_bound = -lb_open
        if model.slim_optimize(error_value=0) > TOL:
            return "ok", []
        dead = [m for m, c in bio.metabolites.items() if c < 0 and len(m.reactions) == 1]
        with model:
            for m in dead:
                model.add_boundary(m, type="sink", lb=-1.0)
            step1_ok = model.slim_optimize(error_value=0) > TOL
        if step1_ok:
            sinks = [m.id for m in dead]
        else:
            prec = [m for m, c in bio.metabolites.items() if c < 0]
            order = [m for m in prec if m not in dead] + dead
            with model:
                sk = {m.id: model.add_boundary(m, type="sink", reaction_id=f"SKTMP_{m.id}",
                                               lb=-1.0) for m in order}
                if model.slim_optimize(error_value=0) <= TOL:
                    return "unrepairable", []
                for m in order:
                    sk[m.id].lower_bound = 0.0
                    if model.slim_optimize(error_value=0) <= TOL:
                        sk[m.id].lower_bound = -1.0
                sinks = [mid for mid, r in sk.items() if r.lower_bound < 0]
    for mid in sinks:
        model.add_boundary(model.metabolites.get_by_id(mid), type="sink", lb=-1.0)
    return "repaired", sinks


def parse_pairs(text, default=1000.0):
    """'a:1,b:2' -> {'a': 1.0, 'b': 2.0}; 'a,b' -> default bound."""
    out = {}
    for item in filter(None, (x.strip() for x in text.split(","))):
        k, _, v = item.partition(":")
        out[k.strip()] = float(v) if v else default
    return out


def solver_name(interface):
    """'optlang.cplex_interface' -> 'cplex'."""
    return interface.__name__.split(".")[-1].replace("_interface", "")


def set_solver(name="auto"):
    """Select the LP solver for every model loaded afterwards and return its name.
    auto = cobrapy default: the first installed of gurobi, cplex, glpk.
    Raises an error if the named solver is not installed."""
    if name and name != "auto":
        cobra.Configuration().solver = name
    return solver_name(cobra.Configuration().solver)


def single_thread(model):
    """CPLEX solves each LP with all cores of the node by default (threads = 0);
    with one LP per worker process this oversubscribes the CPUs, so use 1 thread.
    model.copy() does not keep this setting: call it again after every copy."""
    if solver_name(model.solver.interface) == "cplex":
        model.solver.problem.parameters.threads.set(1)
    return model


def load(path):
    return single_thread(cobra.io.read_sbml_model(path))
