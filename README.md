# hpc_random_media

Random-nutrient FBA/pFBA simulations of single-strain genome-scale metabolic models for AI training data (GEMs-AI project, Hildebrand Lab).

For every model, the pipeline simulates many random nutrient conditions. For each condition it records whether the model can produce biomass, which nutrients it takes up and which compounds it secretes. The results serve as training data for an AI model that predicts microbial metabolic activity from nutrient availability.

## Abbreviations

| Abbreviation | Meaning |
|---|---|
| GEM | genome-scale metabolic model (基因组尺度代谢模型) |
| FBA | flux balance analysis (通量平衡分析) |
| pFBA | parsimonious FBA (简约通量平衡分析: FBA optimum with minimal total flux) |
| FVA | flux variability analysis (通量变异性分析) |
| LP | linear programming (线性规划) |
| SBML | Systems Biology Markup Language (系统生物学标记语言, the .xml model format) |
| AGORA / AGORA2 / APOLLO | VMH collections of gut microbial GEMs (肠道微生物代谢模型库) |
| VMH | Virtual Metabolic Human (虚拟人体代谢数据库) |
| HPC | high-performance computing cluster (高性能计算集群) |
| SLURM | job scheduler of the HPC cluster (作业调度系统) |

## 1. Concepts

### 1.1 Exchange reactions and nutrient conditions

A GEM exchanges metabolites with its environment through exchange reactions (`EX_<metabolite>_e`, written `EX_glc_D_e_` or `EX_glc_D(e)` in VMH models). Flux convention:

- negative flux = uptake from the medium
- positive flux = secretion into the medium

The lower bound of an exchange controls uptake. A lower bound of 0 means the compound is absent. A lower bound of -1000 means the compound is present and its uptake is not limited. The upper bound (1000) is never changed, so every compound can be secreted, including compounds that are also supplied.

A nutrient condition is a set of exchanges with lower bound -1000. All other exchanges have lower bound 0.

### 1.2 FBA and pFBA

- FBA maximises the biomass reaction under the condition. Biomass > 1e-6 h-1 counts as growth.
- pFBA keeps the FBA optimum and minimises the sum of absolute fluxes. The pFBA fluxes give the uptake and secretion pattern that is reported.

The pFBA solution is one of several equivalent optimal solutions. A product that is absent in pFBA may still be possible at the same growth rate (FVA would show this; FVA is not part of this pipeline).

### 1.3 Two reasons why a model cannot grow

**Problem A, model defect.** The biomass reaction needs a precursor that the model cannot synthesise under any condition. Example: `Bifidobacterium_breve_DSM_20213_AGORA2.xml` (VMH release 2026-09-18) contains `pe180_c`, `peai17_c`, `pei17_c` and `pg180_c` only in `bio1`. No reaction makes them, so biomass is 0 even with every nutrient supplied. Such models are repaired automatically (section 4.2), because training data must come from a model that can make biomass.

**Problem B, auxotrophy and missing nutrients.** The model needs nutrients that are absent in the condition (vitamins, amino acids, ions, N, P or S sources). This is a property of the strain and of the condition. It is NOT compensated: essential nutrients are sampled randomly like all other nutrients, so the AI can learn when a strain cannot grow.

## 2. Files

```
hpc_random_media/
├── README.md                  this file
├── gemtools.py                shared functions (exchange setup, repair of Problem A)
├── check_model.py             checks one or more models, writes models_report.tsv
├── random_media_pfba.py       stage 1: random-nutrient FBA/pFBA of one model
├── run_random_media.sbatch    stage 1 SLURM array job, one task per model in models/
├── dfba_random_conc.py        stage 2: dFBA with random concentrations (section 10)
├── run_dfba.sbatch            stage 2 SLURM array job, N_CHUNKS tasks per model
├── submit_dfba.sh             submits run_dfba.sbatch (all models x parts) and the merge job
├── run_dfba_merge.sbatch      merge job (runs merge_dfba.py on every model)
├── merge_dfba.py              combines the parts of a split stage-2 run
├── environment.yml            Python environment (python 3.11, cobra, pandas, numpy)
├── LICENSE                    MIT License
├── .gitignore                 keeps models, results and logs out of git
├── models/                    SBML models (.xml), one file per strain
└── logs/                      SLURM logs
```

`gemtools.py` must stay in the same folder as `check_model.py` and `random_media_pfba.py`.

## 3. Step-by-step use on the HPC

### Step 1. Get the code

Clone the repository on the HPC:

```bash
cd ~
git clone https://github.com/Lishijiagg/GEM_for_AI_training.git
cd GEM_for_AI_training
```

Without git access on the HPC, download the repository as a zip on your own computer, copy it with `scp hpc_random_media.zip <user>@<hpc-login-node>:~/` and unzip it there.

### Step 2. Create the Python environment (once)

With micromamba (used on the QIB/NBI HPC):

```bash
micromamba create -f environment.yml
micromamba activate gemai
python3 -c "import cobra; print(cobra.__version__)"
```

With conda the commands are the same with `conda` in place of `micromamba`. The last line must print a version number (tested with cobra 0.32.1).

### Step 3. Put the models into models/

Copy every SBML model to simulate into `models/`. The file name (without `.xml`) becomes the name of the result folder.

Model files are not part of this repository. AGORA2 and APOLLO models are distributed by the Virtual Metabolic Human resource (https://www.vmh.life). The pipeline reads any SBML model with exchange reactions; it was tested with AGORA 1.03 and AGORA2 models (VMH IDs) and with CarveMe models (BiGG IDs).

```bash
cp /path/to/AGORA2/*.xml models/
ls models/
```

### Step 4. Check all models

```bash
python3 check_model.py models/*.xml
```

This takes about 3 to 10 seconds per model and writes `models_report.tsv`:

| Column | Meaning |
|---|---|
| model | file name |
| biomass_reaction | objective reaction (e.g. `bio1`, `biomass345`) |
| status | `ok`: grows with every candidate supplied; `repaired`: grows after sinks for missing biomass precursors were added; `unrepairable`: no biomass even after repair, skipped in the simulation |
| sinks | sinks added for Problem A |
| n_exchanges / n_candidates | all exchanges / exchanges that are sampled as nutrients |
| growth_all_open | biomass with every candidate at -1000 |
| n_essential / essential | candidates whose single removal from the full set stops growth (Problem B; informative only, these stay random) |

Example:

```
model                                       status    sinks                              n_candidates  n_essential  essential
Bifidobacterium_breve_DSM_20213_AGORA2.xml  repaired  pe180_c;peai17_c;pei17_c;pg180_c  155           13           zn2;ca2;cl;fe3;k;ribflv;so4;pnto_R;mg2;cu2;cobalt2;mn2;4abz
```

### Step 5. Quick test (about 10 seconds)

```bash
python3 random_media_pfba.py --model models/Bifidobacterium_breve_DSM_20213_AGORA2.xml \
  --out results/test --n 20
rm -r results/test
```

The last printed line has the form `biomass produced in X% of 20 conditions (dropout X%, subset X%)`.

### Step 6. Submit the simulation of all models

Edit the settings block of `run_random_media.sbatch` if needed (`N`, `FRAC_DROPOUT`, `M_MAX`, `ENV_NAME`; add `#SBATCH -p <partition>` if the cluster requires a partition). Then submit one array task per model:

```bash
micromamba activate gemai
sbatch --array=1-$(ls models/*.xml | wc -l) run_random_media.sbatch
squeue -u $USER
```

Without `--array` only the first model in `models/` is run. Do not add or remove model files while the job runs, because the task number refers to the position in the sorted file list.

### Step 7. Follow the progress

```bash
cat logs/random_media_<jobid>_<task>.out          # environment, model check, final summary
cat logs/random_media_<jobid>_<task>.err          # errors
tail -f results/<model name>/simulation.log       # one line per condition (Ctrl+C to stop viewing)
```

A finished task ends with `== done ==` in the `.out` file. 1,000 conditions take about 1 to 8 minutes per model, depending on the file system.

### Step 8. Check that each model finished

```bash
tail -n 3 logs/random_media_*.out
ls -d results/<model name>/cond_* | wc -l          # must equal N
```

### Step 9. Look at the results

```bash
cd results/<model name>
cat run_info.txt
column -t -s $'\t' conditions_summary.tsv | head
column -t -s $'\t' cond_0008/medium.tsv | head -20
column -t -s $'\t' cond_0008/products.tsv
```

Use `column -t -s $'\t'` (split on tabs); plain `column -t` misaligns names that contain spaces.

Product frequencies across growing conditions:

```python
import pandas as pd
P = pd.read_csv("products_presence.tsv", sep="\t", index_col=0)
grow = P[P.biomass_produced == 1].drop(columns="biomass_produced")
print(len(grow), "of", len(P), "conditions grow")
print((grow.mean() * 100).round(1).sort_values(ascending=False).head(20))
```

To open the tables in Excel, copy the result folder to your computer:

```bash
scp -r <user>@<hpc-login-node>:~/hpc_random_media/results/<model name> .
```

## 4. Simulation design

### 4.1 Exchange roles

| Role | Compounds | Lower bound |
|---|---|---|
| fixed | `h2o`, `h` (solvent and proton balance, not nutrients) | -1000 in every condition |
| excluded | `o2` (anaerobic gut conditions) | 0 in every condition |
| candidate | every other exchange of the model: carbon sources, amino acids, peptides, vitamins, ions, NH4, phosphate, sulfate, bile acids, drugs | -1000 if present in the condition, 0 if absent |

Fixed and excluded compounds are set with `--fixed` and `--exclude`. Candidates come from each model's own exchange reactions, so different models have different candidate lists (`candidate_substrates.tsv`).

### 4.2 Repair of Problem A (gemtools.repair_biomass)

1. All candidates are opened at -1000. If biomass > 0, status is `ok` and nothing is changed.
2. Otherwise, a sink reaction (lower bound -1) is added for every biomass precursor that occurs in no reaction except the biomass reaction. If biomass > 0, status is `repaired`.
3. Otherwise, sinks are added for all biomass precursors and then removed one by one. Only the sinks without which biomass is 0 are kept (dead-end precursors from step 2 are tested last, so they are kept preferentially). If biomass is still 0, status is `unrepairable` and the model is skipped.

A sink supplies a cytosolic metabolite directly. In step 2 the metabolite can only flow into the biomass reaction, so the sink cannot act as a carbon or energy source. Step 3 sinks can in principle be used by other reactions; check `models_report.tsv` and `run_info.txt` for models repaired in step 3.

Step 3 was tested by knocking out single branched-chain fatty acid reactions in the B. thetaiotaomicron model (e.g. `3HAD11M12`); the repair kept the downstream lipid sinks (`pei17`, `pgi17`).

### 4.3 Sampling of conditions

Each condition uses one of two designs. `--frac-dropout` sets the share of dropout conditions (default 0.5); designs are shuffled.

| Design | Procedure | Purpose |
|---|---|---|
| subset | r ~ Uniform(0, 1); each candidate is present with probability r | sparse to rich media |
| dropout | all candidates present; m ~ Uniform{1, ..., m_max} randomly chosen candidates removed (`--m-max`, default 20) | rich media with a few missing nutrients; shows which missing nutrients stop growth |

Reason for the dropout design: 11 to 15 candidates are jointly essential in the tested models (mostly ions), so random subsets alone give only 5 to 7% growing conditions.

The random generator uses `--seed` (default 42). The same model, seed and parameters give identical conditions and results.

### 4.4 Per condition

1. Set lower bound -1000 for present candidates (0 for absent ones).
2. FBA: maximise the biomass reaction; record status and biomass.
3. If biomass > 1e-6: pFBA for fluxes. Otherwise all fluxes are 0.
4. Write the condition folder and timing.

Solver: the default cobrapy solver (GLPK unless another solver is installed).

## 5. Output reference

```
results/<model name>/
├── run_info.txt               parameters, model check status, added sinks
├── simulation.log             start line, one line per condition, final timing summary
├── candidate_substrates.tsv   candidate exchanges (exchange_id, metabolite, name, role)
├── conditions_summary.tsv     one row per condition
├── substrates_presence.tsv    condition x candidate nutrient (1 present, 0 absent) + biomass_produced
├── products_presence.tsv      condition x secreted compound (1 secreted, 0 not) + biomass_produced
└── cond_0001/ ... cond_N/
    ├── result.tsv             the condition's row of conditions_summary.tsv
    ├── medium.tsv             present nutrients: exchange_id, metabolite, name, lower_bound, flux
    ├── products.tsv           exchanges with flux > 0, sorted by flux
    └── exchanges.tsv          all exchanges: role, lower_bound, upper_bound, flux
```

### conditions_summary.tsv / result.tsv

| Column | Meaning |
|---|---|
| condition | `cond_0001` ... |
| design | `subset` or `dropout` |
| status | LP status (`optimal` expected) |
| biomass_reaction, biomass | objective reaction and its FBA value (h-1) |
| n_substrates_set | number of candidates present |
| n_substrates_used | present candidates with uptake flux < 0 in pFBA |
| n_products | exchanges with secretion flux > 0 (includes h2o) |
| fba_s, pfba_s, write_s, total_s | time in seconds for FBA, pFBA (about 0 without growth), writing tables, whole condition |

### medium.tsv

`flux` < 0: taken up; 0: present but not used; > 0: net secreted although supplied.

### exchanges.tsv

`role` is `substrate_set` (present candidate), `candidate` (absent candidate), `fixed` or `excluded`.

### substrates_presence.tsv and products_presence.tsv

One row per condition, first column `biomass_produced` (1 = biomass > 1e-6). `products_presence.tsv` lists every compound with net secretion in pFBA in at least one condition, except `h2o` and `h`. A compound counts as a product also when it was supplied in the same condition. These two tables are the main AI input/label tables.

## 6. Interpretation and caveats

1. **Biomass values only mean yes/no.** With every present nutrient at -1000, growth is not limited by any nutrient. It is limited by the default reaction bounds of the model (±1000): in B. breve condition `cond_0008`, central reactions such as GAPD, ENO, PGK and ATPS4 run at exactly 1000 and biomass is 50.8 h-1. Raising all exchange upper bounds to 10000 left biomass unchanged; raising all internal reaction bounds to 10000 raised it to 116.5 h-1. Biomass is about 51 h-1 in every growing B. breve condition. Do not compare biomass values between conditions.
2. **Flux values are not physiological.** For the same reason, uptake and secretion magnitudes reflect the 1000 bounds (several products are secreted at exactly 1000). Use the presence/absence of uptake and secretion.
3. **Overflow products.** Unlimited uptake lets the model secrete sugars and amino acids released from supplied oligosaccharides, peptides and polysaccharides (e.g. galactose and fructose in 100% of growing B. breve conditions).
4. **pFBA is one of several optimal solutions.** Absence of a product in pFBA does not prove that it cannot be made.
5. **Supplied and secreted.** A supplied compound can be net secreted (e.g. ethanol in B. breve is never taken up and often secreted).
6. **Ions dominate no-growth.** Most no-growth conditions lack one of the essential ions. All 260 no-growth dropout conditions of B. breve lacked at least one of its 13 essential candidates.
7. **Drugs are candidates.** AGORA2 drug-metabolism exchanges (clonazepam, nitrazepam, 5-fluorouracil, chloramphenicol, ...) are sampled like nutrients; their conversion products appear among the products.
8. **Sinks are part of the model.** Repaired models carry the added sinks; they are listed in `run_info.txt`.

## 7. Example results (1,000 conditions, seed 42, default settings)

| Model | Source | Status | Candidates | Essential candidates | Growth (all) | dropout | subset |
|---|---|---|---|---|---|---|---|
| Bifidobacterium breve DSM 20213 | AGORA2 | repaired (4 lipid sinks) | 155 | 13: 9 ions, so4, riboflavin, pantothenate, 4-aminobenzoate | 27.3% | 48.0% | 6.6% |
| Bacteroides thetaiotaomicron VPI-5482 | AGORA 1.03 | ok | 201 | 11: 10 ions, phosphate | 32.7% | 58.0% | 7.4% |
| Faecalibacterium prausnitzii A2-165 | AGORA 1.03 | ok | 155 | 15: 10 ions, folate, nicotinate, pyridoxal, thiamin, tryptophan | 22.4% | 39.6% | 5.2% |

Run time: 0.05 to 0.09 s per condition on a single core in a cloud test; about 0.4 s per condition on the HPC home file system, where writing files took most of the time.

## 8. Parameters

### random_media_pfba.py

| Option | Default | Meaning |
|---|---|---|
| `--model` | required | SBML file |
| `--out` | required | output folder |
| `--n` | 1000 | number of conditions |
| `--frac-dropout` | 0.5 | share of dropout conditions |
| `--m-max` | 20 | dropout design: max number of candidates removed |
| `--lb` | 1000 | uptake bound of a present nutrient (lower bound = -lb) |
| `--seed` | 42 | random seed |
| `--fixed` | `h2o:1000,h:1000` | always-open compounds, `metabolite:bound` (metabolite ID without `_e`) |
| `--exclude` | `o2` | never-supplied compounds; `--exclude ""` makes o2 a candidate |

### check_model.py

`python3 check_model.py <model.xml> [<model.xml> ...] [--report models_report.tsv] [--fixed ...] [--exclude ...]`

Use the same `--fixed` and `--exclude` as in the simulation.

### run_random_media.sbatch

Settings block: `N`, `FRAC_DROPOUT`, `M_MAX`, `ENV_NAME`. Resources: 1 CPU, 4 GB, 2 h per model. Output folder: `results/<model file name without .xml>`. The script activates micromamba (found through `$MAMBA_EXE`, `PATH`, `~/.local/bin`, `~/bin` or `~/micromamba/bin`) and stops with an error if cobra cannot be imported.

## 9. Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `conda: command not found` / `micromamba not found` in `.err` | environment manager not available on the compute node | activate the environment before `sbatch`; the job then uses the submitting shell's python |
| `ERROR: cobra not importable` | wrong or missing environment | `micromamba activate gemai` before `sbatch`; check `ENV_NAME` |
| `argument --n: invalid int value: ''` | the python command was typed in the terminal with `$N` etc., which only exist inside the sbatch script | run `sbatch run_random_media.sbatch`, or give the values directly |
| `model cannot produce biomass; skipped` | status `unrepairable` | check the model (e.g. compare with the .mat version from VMH) |
| `ModuleNotFoundError: gemtools` | `gemtools.py` missing | keep it in the same folder as the scripts |
| misaligned columns in the terminal | names with spaces | `column -t -s $'\t' file.tsv` |

## 10. Stage 2: dynamic FBA in random virtual media (dfba_random_conc.py)

### 10.1 Purpose and input

Stage 2 takes the substrate combinations that produced biomass in stage 1 and simulates batch growth over time with random initial concentrations. Input is the stage-1 result folder of the same model (`results/<model name>/substrates_presence.tsv`).

### 10.2 Procedure

For each growing combination, `--n-conc` concentration sets are drawn (default 10). Every nutrient present in the combination gets an initial concentration ~ log-uniform(`--conc-min`, `--conc-max`) mM (default 0.01 to 10 mM); absent nutrients start at 0. Batch culture of 1 L, initial biomass `--x0` (default 0.01 gDW L-1), simulated time `--t-end` (default 24 h).

Uptake is limited without per-compound kinetic parameters (no Vmax, no Km). One global constraint limits total carbon uptake: sum_i nC_i * uptake_i <= `--carbon-cap` (default 60 mmol C gDW-1 h-1, where nC_i is the number of carbon atoms of compound i; 0 switches it off). Section 10.5 explains why this cap is used.

The ODEs are integrated every `--dt` (default 0.01 h). The state is written to the tables every `--record-dt` (default 0.1 h), which keeps output size manageable.

At each time step (dt):

1. Uptake limit from the medium:
   `lower bound of EX_i = -C_i / (X * dt)` (mmol gDW-1 h-1; C_i in mM, X in gDW L-1, dt in h). The cells can take up at most what the medium holds during this step. Compounds secreted earlier enter the medium and can be taken up later. h2o and h stay open, o2 stays closed.
2. FBA maximises biomass (subject to the carbon cap) and gives the growth rate mu. Fluxes come from FBA or pFBA; both are run for every concentration set (`--methods fba,pfba`).
3. Explicit Euler update of the ODEs dX/dt = mu X and dC_i/dt = v_i X:
   `C_i(t+dt) = C_i(t) + v_i X(t) dt`, `X(t+dt) = X(t) + mu X(t) dt`. With the uptake limit above, concentrations never become negative.
4. When mu = 0, nothing changes any more (no forced maintenance flux); the last state is kept until `--t-end`.

Model repair (Problem A) is the same as in stage 1.

`max_growth_rate`, `time_max_growth` and `stop_time` in the summary are taken from every integration step, not only from the recorded time points.

### 10.3 Use on the HPC

Stage 1 must be finished for every model first.

Simulation settings are in the settings block of `run_dfba.sbatch` (defaults): N_CONC=10, CONC_MIN=0.01, CONC_MAX=10, X0=0.01, T_END=24, DT=0.01, RECORD_DT=0.1, METHODS=fba,pfba, CARBON_CAP=60. Resources per array task are in its `#SBATCH` lines (default 8 CPUs, 8 GB, 24 h).

The concentration sets of each model are split into `N_CHUNKS` parts; each part is one array task. Settings in `submit_dfba.sh`: `N_CHUNKS` (default 10) and `MAX_RUNNING` (maximum number of tasks running at the same time, 0 = no limit). Submit:

```bash
micromamba activate gemai
bash submit_dfba.sh
squeue -u $USER
```

`submit_dfba.sh` submits M models x N_CHUNKS tasks (task t runs model ((t-1) / N_CHUNKS) + 1, part ((t-1) % N_CHUNKS) + 1) and a merge job with `--dependency=afterany`, which starts when all tasks have ended. Example output:

```
submitted dFBA job 24620000: 2 models x 10 parts = 20 tasks (array 1-20)
submitted merge job 24620001 (starts after all tasks of 24620000 have ended)
```

Progress of one part: `tail -f results_dfba/<model name>/dfba_chunk001of010.log`.

Part i takes every N_CHUNKS-th concentration set (i, i + N_CHUNKS, ...). All concentration sets are drawn before splitting, so the sets and the results do not depend on N_CHUNKS. Test: B. breve, 3 combinations x 2 sets, FBA, split into 3 parts and merged vs. one part: identical summary, initial concentrations and net changes.

While parts run, each writes `run_info_chunkIIIofNNN.txt`, `dfba_chunkIIIofNNN.log`, `dfba_summary_chunkIIIofNNN.tsv`, `dfba_initial_conc_chunkIIIofNNN.tsv` and `dfba_net_change_chunkIIIofNNN.tsv`; the condition folders are shared. The merge job checks that all parts finished, writes the five files of section 10.4 and moves the part files to `chunks/`. If a part failed, the merge job reports the missing parts and merges nothing. Re-submit only the failed task and merge by hand:

```bash
sbatch --array=<task number> --export=ALL,N_CHUNKS=10 run_dfba.sbatch
python3 merge_dfba.py results_dfba/<model name>
```

Without splitting (one task per model): `sbatch --array=1-$(ls models/*.xml | wc -l) run_dfba.sbatch` (N_CHUNKS defaults to 1, files without suffix, no merge needed).

Expected load with the defaults (from a 4-run test): about 66 CPU-seconds per concentration set (FBA about 8 s, pFBA about 58 s) and about 0.75 MB of output per set. B. breve DSM 20213 (273 growing combinations, 2,730 sets): about 50 CPU-hours, about 3 h on 16 CPUs at the test speed, about 2 GB. Quick test of one model:

```bash
python3 dfba_random_conc.py --model models/<model>.xml --stage1 results/<model> \
  --out results_dfba/test --max-conditions 3 --n-conc 2
```

### 10.4 Output (results_dfba/<model name>/)

| File | Content |
|---|---|
| `run_info.txt` | parameters |
| `dfba.log` | one line per run (condition, concentration set, method, final biomass, max growth rate, stop time, timing) |
| `dfba_summary.tsv` | one row per run x method: initial and final biomass, max growth rate, time of max growth, stop time, number of nutrients, consumed and produced compounds, timing |
| `dfba_initial_conc.tsv` | one row per run: initial concentration (mM) of every nutrient (AI input) |
| `dfba_net_change.tsv` | one row per run x method: final minus initial concentration (mM) of every compound that changed (negative = consumed, positive = produced) |
| `cond_XXXX/conc_YY/initial_medium.tsv` | nutrients and initial concentrations of this run |
| `cond_XXXX/conc_YY/<method>/timecourse.tsv` | time, biomass (gDW L-1), growth rate (h-1), every record_dt |
| `cond_XXXX/conc_YY/<method>/concentrations.tsv` | time x compound (mM), compounds that are non-zero at some time |
| `cond_XXXX/conc_YY/<method>/fluxes.tsv` | time x exchange (mmol gDW-1 h-1), exchanges with non-zero flux at some time |
| `cond_XXXX/conc_YY/<method>/summary.tsv` | the run's row of dfba_summary.tsv |

### 10.5 Why the carbon cap (tests on B. breve DSM 20213)

Test: stage-1 condition `cond_0008`, first concentration set, FBA, x0 = 0.01 gDW L-1.

- Without kinetic limits, the growth rate is set by the model's default reaction bounds (±1000). The first step had mu = 46.9 h-1; with all internal bounds raised to 1e5, mu rose to 106 h-1. Growth stops within 0.1 to 0.3 h once an essential nutrient is used up, so the 24 h time course is constant after that.
- Final biomass is set by the nutrient amounts and does not depend on dt (0.5782 gDW L-1 for dt = 0.1, 0.01 and 0.001 h).
- Product amounts depend on dt when growth is this fast. Acetate: 37.0 mM at dt = 0.1 h, 13.3 mM at 0.01 h, 13.4 mM at 0.001 h; ethanol 32.6, 13.3, 13.4 mM. dt <= 0.01 h gives converged end points.
- With `--carbon-cap 60` the same run had mu = 1.66 h-1, growth stopped at 6.5 to 6.7 h, final biomass was again 0.5782 gDW L-1, and the products did not depend on dt (acetate 6.81, formate 12.55, ethanol 4.59 mM at dt = 0.1 and 0.01 h). The product profile differs from the run without cap.
- Run time (2 CPUs, 20 combinations x 2 sets x 2 methods, dt = 0.1 h): 14 s without cap, 210 s with cap.
- Decision (2026-11-05): carbon cap 60 mmol C gDW-1 h-1, dt = 0.01 h, 24 h. Check run (4 combinations x 1 set): growth rate 1.66 h-1 at the start, growth stopped at 2.5 to 6.7 h, final biomass identical to the dt = 0.1 h run (cond_0008: 0.5782 gDW L-1).

### 10.6 Parallel execution

- Across models and parts: each model is split into `N_CHUNKS` array tasks (section 10.3); all tasks can run at the same time on different nodes.
- Within a model: the concentration sets are independent and run in parallel with `--processes` worker processes (`run_dfba.sbatch` passes `SLURM_CPUS_PER_TASK`, default 8). Each worker loads the model once and takes the next set as soon as it finishes one (one set per task).
- Within one concentration set the time steps depend on each other and run one after another; FBA and pFBA of the same set run one after the other in the same worker.
- Every run uses a fresh copy of the model. Without this, the LP solver starts from the basis of the previous run in the same worker, and with alternative optima the FBA trajectory depended on the run order: one test set gave a final biomass of 0.555997 gDW L-1 with 1 process and 0.555497 with 2 processes. With fresh copies the results of 1 and 2 processes were identical (final biomass and all net changes, 4 sets x 2 methods). Copying takes about 0.17 s per run.
- Speed-up in the cloud test (4 sets, FBA + pFBA): 256 s with 1 process, 139 s with 2 processes.
- Stage 1 (`random_media_pfba.py`) runs the conditions of one model one after another on one CPU; at about 0.05 to 0.4 s per condition it needs no further parallelisation.

## 11. Version history

- v1 (2026-11-02): random substrate subsets with random lower bounds (-10 to -0.1), fixed inorganic background.
- v2 (2026-11-03): presence/absence design with lower bound -1000; manual `--extra-background` for auxotrophies; automatic sinks for dead-end biomass precursors.
- v5 (2026-11-05): stage 2 split into N_CHUNKS array tasks per model (submit_dfba.sh, run_dfba_merge.sbatch, merge_dfba.py).
- v4 (2026-11-05): stage 2 added, dFBA with random initial concentrations (dfba_random_conc.py, run_dfba.sbatch); defaults carbon cap 60 mmol C gDW-1 h-1, dt 0.01 h, record_dt 0.1 h; one concentration set per worker task and a fresh model copy per run (results independent of --processes).
- v3 (2026-11-03): Problem A repaired automatically in three steps; Problem B not compensated, all nutrients including ions, NH4, phosphate, sulfate and vitamins are random candidates; subset + dropout sampling; batch model check; SLURM array over all models.

## 12. License

MIT License, see `LICENSE`.
