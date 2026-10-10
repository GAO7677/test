# Exact context/future sigma experiment

This separate experiment uses the three pinned `historical/*_noisy.py` demo configurations: gap038, N75, Wan step1000, Nano/Predict2.5 step1500. It replaces the old `nearest_train(0.75 * future_sigma)` context schedule with:

```text
context_sigma = future_sigma
context_timestep = future_timestep
context_input = (1 - context_sigma) * clean_context + context_sigma * fixed_context_noise
```

Both equalities are exact, without nearest-grid quantization. In particular, do not compute `context_timestep = 1000 * future_sigma` when the native scheduler uses an integer timestep: copy its actual timestep. The checkpoint, input, future RNG, context RNG, CFG, mask and native sampler are retained. Wan still uses the context slice of full initial noise; Nano and Predict2.5 still use independent fixed FP32 context noise with seed20261007. There is no RePaint resampling or fresh context noise per step.

`run.py` validates the pinned historical assets and runtime. Wan uses the existing exact scale=1 endpoint. Nano/Predict2.5 use narrowly checked source substitutions; the executed source and its hashes are archived under the output directory. The original six historical entries and their frozen code are unchanged. Each run writes `experiment_plan.json`, `sigma_trace.json`, and `experiment_verification.json`; PASS means sigma/time alignment passed, not byte-exact historical reproduction.

Run from the original PhysRVG workspace with all external dependencies described in [the historical README](../historical/README.md). Use a new output directory for every run. Examples:

```bash
/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python scripts_mytrain/inference/equal_sigma/run.py --backbone wan --gpu 2 --output /data/gaoya/agent-data/outputs/equal_sigma_20261010/wan
/data/gaoya/agent-data/envs/cosmos_nano_acn75/bin/python scripts_mytrain/inference/equal_sigma/run.py --backbone nano --gpu 3 --output /data/gaoya/agent-data/outputs/equal_sigma_20261010/nano
(
  source scripts_mytrain/inference/historical/frozen/predict25/env.sh
  /data/gaoya/agent-data/envs/cosmos_predict25_official/bin/python scripts_mytrain/inference/equal_sigma/run.py --backbone predict25 --gpu 2 --output /data/gaoya/agent-data/outputs/equal_sigma_20261010/predict25
)
```

The demos are a within-backbone comparison. Wan and Predict2.5 retain the old Test70 sources, whereas Nano uses Test70_v2, so the inputs are not a controlled cross-backbone comparison. Predict2.5's six-entry historical replay limitation remains documented in its original verification report.

The nine-video gallery is [comparison.html](../../../demos/gap038/comparison.html). To serve the published repository in the foreground:

```bash
python3 -m http.server 8845 --bind 0.0.0.0 --directory /data/gaoya/agent-data/outputs/github-context-inference-20261010-efc_zrlo/demos/gap038
```

Open `http://localhost:8845/comparison.html` on that machine, or `http://10.176.42.45:8845/comparison.html` on its network.
