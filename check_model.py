#!/usr/bin/env python3
"""
check_model.py
Check one or more GEMs before random_media_pfba.py, using the same rules
(only h2o and h fixed, o2 excluded, every other exchange is a candidate).

  python3 check_model.py models/*.xml                 # writes models_report.tsv
  python3 check_model.py models/*.xml --report my_report.tsv

Per model
  status       ok           : biomass with all candidates open, no repair needed
               repaired     : sinks added for biomass precursors without a synthesis route
               unrepairable : no biomass even after repair; not usable
  sinks        precursor sinks added (Problem A, model defect)
  growth_all_open   biomass with every candidate at -1000
  essential    candidates whose single removal from the full set stops growth
               (Problem B: auxotrophies and required inorganic nutrients;
               these stay random in the simulation)
"""
import argparse
import logging
import os

import pandas as pd

import gemtools as gt

logging.getLogger("cobra").setLevel(logging.ERROR)


def check(path, fixed, exclude, lb=1000.0):
    model = gt.load(path)
    biomass = gt.biomass_id(model)
    table, candidates = gt.setup_exchanges(model, fixed, exclude)
    status, sinks = gt.repair_biomass(model, candidates, lb)
    row = dict(model=os.path.basename(path), biomass_reaction=biomass, status=status,
               sinks=";".join(sinks), n_exchanges=len(table), n_candidates=len(candidates))
    if status == "unrepairable":
        return row
    name = {rid: b for rid, b, _ in table}
    for rid in candidates:
        model.reactions.get_by_id(rid).lower_bound = -lb
    row["growth_all_open"] = round(model.slim_optimize(error_value=0), 4)
    ess = []
    for rid in candidates:
        r = model.reactions.get_by_id(rid)
        r.lower_bound = 0.0
        if model.slim_optimize(error_value=0) <= gt.TOL:
            ess.append(name[rid])
        r.lower_bound = -lb
    row["n_essential"] = len(ess)
    row["essential"] = ";".join(ess)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="SBML files")
    ap.add_argument("--report", default="models_report.tsv")
    ap.add_argument("--fixed", default="h2o:1000,h:1000")
    ap.add_argument("--exclude", default="o2")
    ap.add_argument("--solver", default="auto",
                    help="LP solver: auto (gurobi > cplex > glpk, first installed), cplex, glpk, gurobi")
    a = ap.parse_args()
    fixed = gt.parse_pairs(a.fixed)
    exclude = set(filter(None, (x.strip() for x in a.exclude.split(","))))
    print(f"solver: {gt.set_solver(a.solver)}")
    rows = []
    for p in a.models:
        row = check(p, fixed, exclude)
        rows.append(row)
        print(f"{row['model']}: {row['status']}"
              + (f"; sinks {row['sinks']}" if row["sinks"] else "")
              + (f"; growth all open {row['growth_all_open']}; "
                 f"{row['n_essential']} essential: {row['essential']}"
                 if "essential" in row else ""))
    pd.DataFrame(rows).to_csv(a.report, sep="\t", index=False)
    print(f"report written to {a.report}")


if __name__ == "__main__":
    main()
