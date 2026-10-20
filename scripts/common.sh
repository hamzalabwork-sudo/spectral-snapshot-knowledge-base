#!/bin/bash
# Shared settings for the run scripts. Run every script from the repository root, e.g.
#   bash scripts/run_benchmark.sh order_A ifr
# Set PYTHON to choose the interpreter (default: python) and CUDA_VISIBLE_DEVICES to choose the GPU.
set -euo pipefail
PY=${PYTHON:-python}
BENCHMARKS="electricity traffic ettm1 ettm2 weather order_A order_B order_C"
mkdir -p logs

# bench_config <benchmark>  ->  "<config> <data>"   (cross-domain: data is the folder with all CSVs)
bench_config() {
  case $1 in
    electricity) echo "configs/within/electricity.yaml data/electricity.csv" ;;
    traffic)     echo "configs/within/traffic.yaml data/traffic.csv" ;;
    ettm1)       echo "configs/within/ettm1.yaml data/ETTm1.csv" ;;
    ettm2)       echo "configs/within/ettm2.yaml data/ETTm2.csv" ;;
    weather)     echo "configs/within/weather.yaml data/weather.csv" ;;
    order_A|order_B|order_C) echo "configs/crossdomain/crossdomain_${1#order_}.yaml data" ;;
    *) echo "unknown benchmark: $1" >&2; return 1 ;;
  esac
}

# bench_seeds <benchmark>  ->  test-split seeds (five within a dataset, ten across domains)
bench_seeds() { case $1 in order_*) echo "0 1 2 3 4 5 6 7 8 9" ;; *) echo "0 1 2 3 4" ;; esac; }

# bench_group <benchmark>  ->  result folder of the benchmark's test runs
bench_group() { case $1 in order_*) echo "crossdomain" ;; *) echo "main" ;; esac; }

# The 16 methods of the benchmark (run_continual.py names).
METHODS="finetune ewc si mas agem ifr_uniform mir derpp clser adaptivecl fsnet fredf clfd clfd_ifr spectracl ifr"
