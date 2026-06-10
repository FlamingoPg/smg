# LLM load-balancing bake-off — results & conclusions

Framework: `sim.py` (event-driven, iteration-level continuous-batching LLM serving model).
Metric: **queue-delay.p99** = `t_admit − arrival` (the pure routing-controlled latency; excludes
intrinsic prefill+decode). Each cell = mean of 5 seeds; capacity = the queueing knee (round-robin,
queue-delay.p99 < 0.5 s); load swept at ρ ∈ {0.85, 0.95, 1.05} of that knee. Theory in `THEORY.md`.

## Model validation (must pass before trusting any ranking) — `python3 sim.py validate`
All pass: prefill **super-linear** in input (256→16k tok: 14→1772 ms, 127× vs 62× linear);
TPOT rises with **batch** (6→40 ms) and with **context length** (8.6→84 ms, KV attention);
queue-delay **hockey-stick** at the knee (11 ms → 3.2 s); **preemptions only under overload**.

## Three bugs that skepticism/validation caught (and the fixes)
1. **Capacity by "completion ratio" was wrong** — e2e is multi-second, so a finite window always
   leaves a tail unfinished → it under-found the knee. Fix: define capacity by the **queue-delay knee**.
2. **TTFT.p99 is floored by intrinsic prefill** (a 16k prompt = 1.8 s) → masks the routing signal and
   collapsed all heavy-shape capacities to ~2 req/s. Fix: rank on **queue-delay** (`t_admit−arrival`).
3. **My own `EW` had a units bug** — `(tokens)/speed` is *seconds* (~0.1) but the barrier `2k/(1−k)` is
   *dimensionless* (~2), so the barrier dominated and EW ignored the queue (chat: 129 ms vs 64). Fix:
   **`EWQ`** with both terms in seconds → chat 30 ms (matches the best).

## Headline results (queue-delay.p99, ms)

`chat` (homogeneous, short) — ρ=0.85 / 0.95:
```
halo-tok 28/39   EWQ 30/42   lqt 29/37   jsq+if 64/76   pow2+if 64/73   round_robin 44/55   jsq(stale) 280/283   EW(buggy) 129/241
```
`mixed` (80% chat + 20% rag) — ρ=0.85 / 0.95 / 1.05:
```
EWQ 6/6/6   halo-tok 6/6/6   least_kv 9/8/12   lutil 8/6/10        <-- KV/token-aware
jsq+if 166/273/332   round_robin 454/491/527   pow2+if 648/484/520   lqt 858/706/650   <-- count- or queue-only
```

## Conclusions (robust, significance-checked)
1. **KV/token-aware routing beats request-count routing by 20–90× when request sizes vary**
   (`mixed`: 6 ms vs 166–648 ms — far beyond seed noise). This is the dominant effect, and matches
   the throughput-optimality of least-*work* over least-*count* (THEORY §4).
2. **The convex KV barrier `k/(1−k)` is necessary.** `lqt` (token-queue *without* barrier) is among the
   *worst* on `mixed` (858 ms); the same token-queue *with* the barrier (`halo-tok`,`EWQ`) is the best
   (6 ms). Matches the M/M/1 congestion derivation (THEORY §3).
3. **In-flight correction is essential under stale polling.** `jsq` (stale, no-if) 280 ms vs `jsq+if`
   64 ms on `chat` — the herding/incast pathology and its fix (THEORY §5).
4. **Unit-consistency matters** (bug #3): the queue term and KV penalty must be in the same units.
5. **Even on homogeneous chat, token-aware ≥ count-aware** (halo-tok/EWQ 28–30 vs jsq+if 64), because
   real "homogeneous" traffic still has output-length variance.
6. **Power-of-2 ≈ full argmin** for the same score (THEORY §2); use pow-2 at fleet scale for O(d) cost.

## Winner: `EWQ` — Expected-Wait, unit-consistent (seconds)
```
EWQ_i = (queued_tokens_i + inflight_tokens_i) / throughput_i   +   λ_t · k_i/(1 − k_i)
        └────────── expected queueing time (s) ──────────┘       └ KV-pressure time (λ_t≈0.15 s) ┘
        argmin over workers (pow-2 sampling at fleet scale); in-flight corrected for stale polls.
```
Overall wins: **EWQ 12**, halo-tok 9, lutil 6, least_kv 5. EWQ is the only score that is simultaneously
best-tier on homogeneous *and* heterogeneous *and* (via `/throughput`) heterogeneous-hardware traffic,
with a principled derivation for every term.

## Caveats / open items (be skeptical)
- Heavy/huge-input shapes (`heavytail`, `bigmix`) calibrate to low req/s, so their p99 is **noisy**
  (few requests/window); the clean signal is from `chat`, `mixed`, `largefleet`. Among KV-aware
  policies (`EWQ`/`halo-tok`/`least_kv`/`lutil`) the differences are mostly within seed noise — they
  are all good; EWQ edges it on count + principled units.
- **Robustness sweep still TODO**: perturb cost coefficients (`c_kv`,`c_attn`,`base` ±2×) and poll
  interval to confirm the winner doesn't flip. Ablation of EWQ's terms (queue / `/μ` / in-flight /
  barrier) to attribute the win is partially done via the contender set (lqt = no barrier loses;
  EW = bad units loses; jsq = count not tokens loses).

---
## Round 2: agentic workloads, ablation, robustness, EWMA (`sim.py ablate|robust|noise`)

**Agentic shape** (16 workers; 50% toolcall + 35% 12k-ctx + 15% 32k-ctx; long, diverse inputs):
KV-aware policies achieve **0 ms** queue-delay (they route around KV-pressured workers); round-robin
152–865 ms, pow2 612–1209 ms. With agentic added, **EWQ wins 17 of all shape×load cells** (most),
then halo-tok 13, least_kv 12, lutil 12.

**Ablation — which EWQ term carries the win (queue-delay.p99 @ρ=0.95):**
| term removed | chat | mixed | agentic | ⇒ the term matters when… |
|---|---|---|---|---|
| full EWQ | 41 | 6 | 0 | — |
| − in-flight | **3067** | 550 | 520 | homogeneous / queueing-dominated (stale herding) |
| − KV barrier | 41 | **943** | **1233** | request sizes vary (avoid KV-stuffed workers) |
| − ÷throughput | 41 | 6 | 0 | (only matters for hetero *hardware*) |
| − tokens (use count) | 60 | 6 | 0 | mild (output-length variance) |
⇒ EWQ = **in-flight correction + KV barrier** are the two load-bearing terms; ÷μ and token-awareness
are refinements for heterogeneous hardware / sizes.

**Robustness — EWQ vs cost-model/poll perturbations (queue-delay.p99 @ρ=0.95):** under `c_kv×2`,
`c_attn×2`, `base×2`, `poll×4`, EWQ stays in the winning tier on chat/mixed/agentic every time
(within ~1 ms of best, or 0), while round-robin is 10–600× worse. EWQ never flips to bad.

**EWMA (the "exponential" question):** EWQ is *Expected-Wait, Queue* — not exponential weighting.
Adding an EWMA-smoothed load (`EWQ-ewma`) under injected load-measurement noise:
| | noise 0.0 | 0.4 | 0.8 |
|---|---|---|---|
| chat: EWQ | 41 | 70 | 77 |
| chat: EWQ-ewma | 41 | **64** | 177 |
| chat: least_kv | 94 | 809 | 1453 |
| chat: jsq+if | 83 | 461 | 1071 |
EWMA helps modestly at moderate noise (64 vs 70) but **hurts at high noise** (lag). The real
noise-robustness comes from EWQ's *structure* (in-flight + barrier + token-queue): EWQ stays ~77 ms
while naive least_kv/jsq+if collapse to 800–1450 ms. On `mixed` the KV signal is so strong that noise
barely matters (EWQ=6 at all noise). ⇒ EWMA is optional, not a core win.

---
## Round 3: prefix-cache / agentic sessions (`sim.py prefix`) — EWQ vs the *full* HALO design

Modeled multi-turn agentic sessions (shared system prompt + growing history = cacheable prefix) with a
per-worker LRU prefix cache; a cache hit skips the cached prefix's prefill. Compared cache-blind load
balancers vs prefix-affinity vs the prod-HALO architecture (affinity + load-gated fallback, EWQ as gate):

| n_sessions=40 (high reuse) | QD.p99 ms | TTFT.p99 ms | cache-hit |
|---|---|---|---|
| cache_affinity (sticky, load-blind) | 557 (worst LB) | 2086 (best) | 55.6% |
| EWQ (load-only, cache-blind) | 202 (best LB) | 3398 | 8.1% |
| **cache+EWQ-gate (HALO arch, EWQ gate)** | 254 | 3374 | 16.5% |

Conclusion (the real EWQ-vs-HALO answer):
1. **As a load score, EWQ > HALO** (token-work vs request-count): ~50× on size-diverse `mixed`.
2. **For agentic prefix-reuse, neither pure-load nor pure-affinity is right** — pure affinity gets the
   best cache-hit/TTFT but the *worst* load balance (it overloads hot-session workers); pure EWQ
   balances best but wastes the cache.
3. **The best system is prod-HALO's *architecture* (prefix-affinity with a load-gated fallback) using
   EWQ as the gate**: it keeps ~EWQ-level load balance while recovering 2-3× the cache hit. So the
   recommendation is not "replace HALO with EWQ" but "use EWQ as the load primitive *inside* the
   affinity+gate router" — i.e. for vLLM PD/agentic, `least_load=EWQ` feeds the cache-aware policy.
