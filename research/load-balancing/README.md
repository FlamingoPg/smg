# Load-balancing research: least-load routing for LLM serving

A discrete-event LLM-serving simulator and algorithm bake-off used to design the
`least_load` routing policy. Conclusion: **EWQ** (Expected-Wait, Queue) —
`argmin_i (queued_tokens_i + inflight_i)/throughput_i + λ_t · k_i/(1−k_i)`,
in-flight-corrected, pow-2 sampled at fleet scale.

## Files
- `sim.py` — event-driven, iteration-level continuous-batching model (finite KV +
  preemption cliff, stale load polling + in-flight, prefix cache). Pure stdlib.
- `THEORY.md` — proofs: JSQ optimality, power-of-d, the `k/(1−k)` KV barrier (M/M/1),
  Lyapunov throughput-optimality of least-work, staleness/herding.
- `RESULTS.md` — validated bake-off + ablation + robustness + agentic/prefix-cache.

## Run
```
python3 sim.py validate   # worker-model fidelity checks (must pass first)
python3 sim.py            # main bake-off (9 shapes x 16 policies x 3 loads x 5 seeds)
python3 sim.py ablate     # which EWQ term carries the win
python3 sim.py robust     # winner stability under cost-model perturbation
python3 sim.py prefix     # agentic multi-turn + prefix-cache (EWQ vs HALO architecture)
```

## Key findings
- KV/token-aware routing beats request-count routing ~20–90× on size-diverse traffic.
- The convex KV barrier `k/(1−k)` and in-flight correction are both necessary (ablation).
- EWQ is the best load *primitive*; for agentic prefix-reuse, use it as the load gate
  inside a prefix-affinity router (prod-HALO architecture).
