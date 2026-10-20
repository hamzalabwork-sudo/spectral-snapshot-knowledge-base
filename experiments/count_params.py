from __future__ import annotations

"""
Total parameter count (trainable and frozen) of every method's forecaster on the profiled
benchmarks, written to results/profile/params_total.json as {bench: {method: count}}.

The profiling runs record only trainable parameters, which hides the encoder that CLFD freezes
after the first regime; this script counts every parameter of the model each method deploys.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_continual import create_model, load_config  # noqa: E402

METHODS = ["finetune", "ewc", "si", "mas", "agem", "ifr_uniform", "mir", "derpp", "clser", "adaptivecl",
           "fsnet", "fredf", "clfd", "clfd_ifr", "spectracl", "ifr"]
BENCHES = {"traffic": ("configs/within/traffic.yaml", 862, [861]),
           "order_A": ("configs/crossdomain/crossdomain_A.yaml", 1, None)}


def main():
    out = {}
    for bench, (cfg_path, d_in, targets) in BENCHES.items():
        cfg = load_config(cfg_path)
        cfg["data"].update(seq_len=96, pred_len=48, target_indices=targets)
        out[bench] = {m: sum(p.numel() for p in create_model(cfg, d_in, 1, "cuda", method=m).parameters())
                      for m in METHODS}
    path = "results/profile/params_total.json"
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
