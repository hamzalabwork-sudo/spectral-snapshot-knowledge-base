# Recovering Forgotten Knowledge in Continual Time-Series Forecasting with a Spectral Snapshot Knowledge Base

Code, configurations and result files for the paper of the same title (manuscript under review).

Continual-learning methods try to prevent forgetting during training and keep a single model, so whatever
that model loses is lost for good. Much of it is not: it survives in the models a continual learner leaves
behind at the end of each regime. On eight benchmarks, an earlier snapshot forecasts an old regime better
than the final model in 82% of (regime, forecast frequency) pairs, and in 85% of old regimes the best
snapshot changes with frequency.

The **spectral snapshot knowledge base (SKB)** recovers this knowledge:

- after every regime it stores the learner's **snapshot** and a small **case segment** (the last 10% of the
  regime's training data, never trained on);
- it records how every snapshot errs at every forecast frequency on the stored cases (cross-snapshot error
  spectra);
- it forecasts with the **convex combination of the snapshots that minimises the expected error separately
  in every frequency bin** of the forecast. No regime label is needed at test time, and the learner itself is
  unchanged, so SKB runs on top of any continual learner.

Selected on validation data and evaluated once on the test split, SKB lowers the test MSE of the strongest
of sixteen methods (a wavelet-encoder learner, CLFD) by 9.4% (7 of 8 benchmarks), reduces its forgetting on
all eight, improves ER, IFR, CLFD and CLFD + IFR, transfers to a longer horizon and a linear backbone, and
keeps most of its gain with three stored snapshots.

## Repository layout

```
run_continual.py         continual training and evaluation; SKB is enabled with --set routing.case_ratio=0.1
spectracl/routing.py     SKB: SnapshotRouter (knowledge base and all compared rules), SpectralStack
                         (deployable SKB module), simplex_weights (exact simplex-constrained solver)
spectracl/               forecasters (Mamba, linear, CLFD encoder) and continual learners (ER, IFR, ...)
data/                    dataset utilities; put the CSV files here (see data/README.md)
configs/                 within-dataset (five seasonal regimes) and cross-domain (three source orders) configs
experiments/             offline rule evaluation, comparison, table and figure scripts
scripts/                 one script per experiment group (see below)
results/                 result files of every run reported in the paper (JSON), and figure caches
```

## Installation

Python 3.10 and an NVIDIA GPU with CUDA 12 are required; the Mamba backbone uses the official CUDA
kernels.

```bash
pip install -r requirements.txt
```

If `mamba-ssm` or `causal-conv1d` fail to build, install the prebuilt wheels that match your PyTorch and
CUDA versions from their GitHub releases (<https://github.com/state-spaces/mamba/releases>,
<https://github.com/Dao-AILab/causal-conv1d/releases>).

## Data

Download the six public CSV files (Electricity, Traffic, ETTm1, ETTm2, ETTh1, Weather) into `data/`, as
described in [`data/README.md`](data/README.md).

## Using SKB

```bash
# CLFD learner with SKB on cross-domain order A, one seed
python run_continual.py --config configs/crossdomain/crossdomain_A.yaml --data data \
    --methods clfd --epochs 20 --seeds 0 --set routing.case_ratio=0.1 --out_dir results/demo
```

With `routing.case_ratio > 0`, the last fraction of every regime's training segment (at least 32 windows)
is held out as a case segment and the run evaluates, on the same snapshots, every rule for using them. The
result file stores each rule's results matrix under `routing` (`sstack_global` is SKB, `final` is the
learner alone), the knowledge base's cost under `profile.skb`, and the snapshot predictions under
`snapshots/*.npz` for offline analysis. The deployable module is `spectracl.routing.SpectralStack`, built by
`SnapshotRouter.deployable()`.

## Reproducing the paper

All scripts run from the repository root; set `PYTHON` to choose the interpreter.

| Paper content | Command | Results |
|---|---|---|
| Baselines (sixteen methods, test split) | `bash scripts/run_benchmark.sh all` | `results/main/`, `results/crossdomain/` |
| Baselines on the validation split (references for the design search) | `bash scripts/run_validation.sh all ifr_uniform ifr clfd clfd_ifr` | `results/val/` |
| Design search: all rules on three learners (validation) | `bash scripts/run_skb_validation.sh all` | `results/val_route/` |
| SKB on four learners (test split, run once) | `bash scripts/run_skb_test.sh all` | `results/test_route/` |
| Size of the case segment (validation) | `bash scripts/run_skb_extra.sh casesize` | `results/val_route_r05/`, `results/val_route_r20/` |
| Horizon 96 and linear backbone (test) | `bash scripts/run_skb_extra.sh h96`, `bash scripts/run_skb_extra.sh linear` | `results/test_h96*/`, `results/test_linear*/` |
| Computational efficiency | `bash scripts/run_profile.sh`, then `bash scripts/run_skb_profile.sh` (idle GPU) | `results/profile/`, `results/profile_skb/` |

Some within-dataset baseline runs were made in an earlier batch and are stored in `results/pilot/`; the table
scripts merge `results/pilot/` and `results/main/`. The baseline runs are those of the companion
paper (Suleiman et al., *Where Do Continual Forecasters Forget?*), whose code and analysis are at
<https://github.com/hamzalabwork-sudo/spectral-intrusion-continual-forecasting>.

**Tables and figures** are regenerated from the stored results without any training:

```bash
bash scripts/make_outputs.sh      # writes outputs/tables/*.tex and outputs/figs/*.pdf
```

| Output | Script |
|---|---|
| `tab_within`, `tab_within_fgt`, `tab_cross`, `tab_overall`, `tab_skbrules`, `tab_skbbases`, `tab_efficiency` | `experiments/make_tables_skb.py` |
| `tab_casesens`, `tab_general`, `tab_search` | `experiments/make_tables_skb_extra.py` |
| `tab_motivation`, `fig_motivation` | `experiments/skb_motivation.py` |
| `fig6_skb_weights` | `experiments/fig_skb_weights.py` |
| memory budget, further design-search rules | `experiments/offline_rules.py` |
| `tab_regimes` (needs the CSV files) | `experiments/make_regime_table.py` |
| paired comparisons quoted in the text | `experiments/compare_route.py` |

The snapshot predictions written by the SKB runs (`results/*/snapshots/*.npz`, about 6 GB) are not part of
the repository. The scripts that need them use the small summaries stored in `results/val_route/offline_*.json`,
`results/test_route/offline_budget.json` and `results/derived/` instead, and recompute them when the
predictions are present.

Results are averaged over five seeds within a dataset and ten across domains; every SKB run is compared
with the official run of the same learner trained on all of its data, paired by seed. GPU kernels are not
fully deterministic, so reruns reproduce the reported numbers up to small run-to-run differences.

## Citation

```bibtex
@unpublished{suleiman2026skb,
  title  = {Recovering Forgotten Knowledge in Continual Time-Series Forecasting with a Spectral Snapshot Knowledge Base},
  author = {Suleiman, Aminu Bashir and Ran, Haowei and Ibrahim, Hamza and Garba, Ahmad Tijjani and Sun, Meijun and Wang, Zheng},
  note   = {Manuscript under review},
  year   = {2026}
}
```

## License

MIT; see [LICENSE](LICENSE). The datasets keep their own licenses.
