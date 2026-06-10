# Least-load routing for LLM serving — theory

Notation. `N` workers. Worker `i` has token-throughput `μ_i` (tok/s), KV capacity
`M_i` (tokens), concurrency cap `C_i`. Requests arrive at rate `λ`; request `r`
has prefill length `p_r` (prompt tokens) and decode length `d_r` (output tokens),
so it injects `p_r + d_r` tokens of work. The router assigns each arrival to a
worker using **stale** load (polled every `T`) plus a local in-flight counter.
Goals: minimize response time (TTFT = queueing + prefill; e2e) and avoid KV
overflow (preemption/recompute). Define per-worker unfinished work (backlog)
`Q_i` (tokens) and KV utilization `k_i = kv_used_i / M_i ∈ [0,1)`.

---

## 1. JSQ is optimal — but only under assumptions LLM serving violates

**Theorem (Winston 1977; Weber 1978).** For `N` *identical* exponential servers,
Poisson arrivals, no jockeying, and **fresh** queue information, Join-Shortest-Queue
minimizes the queue-length vector in the sense of stochastic majorization, and
therefore minimizes mean response time, among all non-anticipating policies that
do not use service-time information.

*Why it does not settle the question.* The proof needs (a) identical servers,
(b) memoryless service, (c) instantaneous queue info. LLM serving breaks all three:

- **Heterogeneous workers** (different GPUs/TP/KV) ⇒ "shortest queue" ≠ "fastest
  service"; you want shortest *work/throughput*, not shortest count.
- **Highly variable service**: a 16k-token prefill is ~400× the work of a 40-token
  decode step. Counting *requests* is a poor proxy for *work*; count token backlog.
- **Stale info** (we poll every `T`): fresh-info optimality says nothing, and naive
  JSQ on stale info actively herds (§6).

So JSQ is the right *idea* (route to least load) but the right *load* is
throughput-normalized token work, measured under staleness — which is what the
proposed `EW` score is (§7). JSQ is our homogeneous-case sanity baseline; the
`jsq_true` oracle (fresh info) is the achievable lower bound the stale policies
are chasing.

---

## 2. Power-of-`d` choices: `d=2` captures (almost) all the benefit

**Theorem (Vvedenskaya–Dobrushin–Karpelevich 1996; Mitzenmacher 2001).** Sample `d`
queues uniformly, join the shortest. As `N→∞` at load `ρ<1`, the stationary tail is
*doubly* exponential:

```
        π_k  ≈  ρ^((d^k − 1)/(d − 1))         (fraction of queues with ≥ k jobs)
```

so the maximum queue length is `≈ ln ln N / ln d + O(1)`.

*Consequences for us.*
- `d=1` (random): max load `~ ln N / ln ln N` — bad tail.
- `d=2`: max load `~ ln ln N` — an exponential improvement over random.
- `d=3` vs `d=2`: only a constant-factor (`1/ln d`) improvement — **diminishing returns**.

Prediction: `pow2 ≈ pow3`, both ≫ random, and on large fleets power-of-`d` rivals
full-argmin at `O(d)` cost instead of `O(N)` — *and* is far more robust to stale
info (§6). This is why we test `EW-pow2`/`EW-d3` alongside full-argmin `EW`.

---

## 3. Why the KV penalty must be the convex barrier `k/(1−k)`

Model a worker's KV as an M/M/1-type resource at utilization `k`. The standing
amount of work an arrival must queue behind is the mean number in system:

```
        L(k)  =  k / (1 − k)            (M/M/1 mean number in system)
        W(k)  =  (1/μ) · 1 / (1 − k)    (mean sojourn time)
```

So the *expected extra delay* from KV contention at occupancy `k` is `∝ k/(1−k)` —
**exactly** HALO's barrier term `λ·k/(1−k)`. Two structural facts make this the
right shape, not a linear `k`:

1. **It is convex and diverges at the cliff.** `d/dk [k/(1−k)] = 1/(1−k)² → ∞` as
   `k→1`. The marginal cost of one more request near KV exhaustion is unbounded —
   which is precisely the preemption/recompute cliff. A linear penalty `α·k` cannot
   represent a cliff; it routes into KV-full workers far too eagerly.
2. **It is dimensionally a "request-equivalent" of memory pressure**, so it adds
   cleanly to a request/work term.

**Therefore any good LLM load score should add a convex-in-`k` barrier.** What
HALO gets *wrong* is the other term (it uses running-*count* as the workload `r`,
ignoring queued tokens and heterogeneity); the barrier itself is principled. The
proposed score keeps the barrier and fixes the workload term (§7).

---

## 4. Least-(token-)work is throughput-optimal — the core case vs RR/random

Consider routing each arrival to `argmin_i Q_i/μ_i` (least *expected drain time*),
i.e. join-shortest-work, throughput-normalized — the `EW` workload term.

**Theorem (Lyapunov drift; Tassiulas–Ephremides MaxWeight 1992).** With Lyapunov
function `V(Q) = Σ_i Q_i² / μ_i`, the one-step drift satisfies

```
        E[ΔV | Q]  ≤  B  −  ε · Σ_i Q_i          for some ε>0
```

whenever the offered work lies strictly inside the capacity region
(`λ·E[p+d] < Σ_i μ_i` with a matching componentwise condition). Negative drift for
large `‖Q‖` ⇒ the chain is positive recurrent ⇒ **all backlogs are stable** (finite
mean). Hence least-token-work routing is *throughput-optimal*: it keeps latency
bounded for **every** load the cluster can serve at all.

*Proof sketch.* `ΔV = Σ_i [(Q_i + A_i − S_i)² − Q_i²]/μ_i
= Σ_i 2 Q_i (A_i − S_i)/μ_i + (bounded)`. The router chooses where arrivals `A_i`
land; sending each arrival to `argmin Q_i/μ_i` *minimizes* `Σ_i Q_i A_i/μ_i`, while
work-conserving service *maximizes* `Σ_i Q_i S_i/μ_i`. Inside the capacity region
there is a static split making `E[S_i−A_i]` uniformly positive on the heavy
coordinates, giving the `−εΣQ_i` term. ∎

*Why this beats round-robin/random.* RR and random are **not** throughput-optimal
under heterogeneity: they split *by request count*, so a slow worker (or one that
drew large requests) accumulates unbounded backlog even while the cluster is below
aggregate capacity — its `Q_i/μ_i` diverges. Least-work self-corrects by routing
away from any worker whose normalized backlog grows. This is the rigorous argument
that a least-(token-)load policy is not just "nicer" but *strictly* extends the
stable load region for heterogeneous / size-skewed workloads.

---

## 5. Staleness ⇒ herding; in-flight correction is mandatory

Let the router poll every `T` and route on the frozen snapshot in between.

**Claim (incast).** Pure `argmin` on a stale snapshot sends *every* arrival in a
poll interval to the single worker `i*` that was least-loaded at the poll: a burst
of `≈ λT` requests onto one worker while the rest idle. Balanced offered load is
converted into worst-case imbalance; the victim's backlog jumps by `λT`, so

```
        p99 inflation  ≈  Θ(λT / μ)   (one interval's arrivals serialized on one worker)
```

This is visible in the sim: stale `jsq`/`halo`/`EW-noif` blow up while their `+if`
versions do not.

**Fix (in-flight correction).** Keep `f_i` = #requests dispatched to `i` since its
last poll, and route by `argmin (ℓ_i + f_i)` (`ℓ_i` = snapshot load). Each placement
increments the effective load, so within an interval the router performs
**water-filling**: it greedily equalizes `ℓ_i + f_i` across workers. For a separable
convex objective this greedy increment is optimal, so after `m = λT` placements the
effective loads are equal to within one unit ⇒ imbalance `O(1)` instead of `O(λT)`,
*without any fresh polling*. In-flight correction recovers (most of) fresh-info JSQ
from stale snapshots.

**Power-of-`d` as a complementary mitigation.** Randomized sampling caps the herd:
an arrival only "sees" `d` random workers, so the probability the whole burst lands
on one worker is `≤ (1/N)^{...}`; expected herd per worker is `O(λT/N)`. In-flight
correction (bias removal) and power-of-`d` (variance/herd capping) are orthogonal;
`EW-pow2` combines both and should be the most robust under bursty+stale load.

---

## 6. The proposed score `EW` (Expected-Wait), and what it unifies

```
        EW_i  =  ( W_i + f_i · p̄ ) / μ_i   +   λ · k_i / (1 − k_i)
                 └─────────────�‐──────┘       └──────────────┘
                 expected queueing (TTFT)        KV cliff barrier
```

- `W_i` = **queued tokens** (token-aware work ahead) — §1 (work not count), §4
  (token-work is throughput-optimal).
- `/μ_i` = throughput normalization — §1/§4 heterogeneity ("least *time*", not least
  *tokens*).
- `f_i · p̄` = **in-flight correction** (`p̄` = mean prefill) — §5 (kills herding under
  stale polls).
- `λ·k_i/(1−k_i)` = **convex KV barrier** — §3 (cliff-aware, M/M/1-principled).
- `argmin` for small fleets; `pow2`/`pow_d` for large fleets — §2.

`EW` is an estimator of the new request's expected TTFT (work-ahead ÷ drain-rate)
plus the KV-cliff penalty, computed from a stale snapshot with in-flight
correction. Predicted behavior, to be confirmed empirically:

| Regime | Prediction | Mechanism |
|---|---|---|
| homogeneous, uniform sizes (`chat`) | `EW ≈ JSQ+if ≈ oracle` | tokens ≈ const ⇒ token-aware = count-aware |
| mixed / heavy-tailed sizes | `EW`, `EW-full` < count policies | token-aware `W_i` |
| heterogeneous workers | `EW`, `wjsq+if` < unnormalized | `/μ_i` term |
| bursty + stale poll | `+if` and `pow2` variants win; `EW-pow2` most robust | §5 |
| KV-saturating (long context) | barrier policies (`EW`,`halo+if`) cut preemptions | §3 |
| 16-worker fleet | `EW-pow2`/`EW-d3` ≈ full `EW` at `O(d)` cost | §2 |

The empirical bake-off (`sim.py`, `RESULTS.md`) tests exactly these predictions
across 9 traffic shapes × 3 utilizations × 3 seeds.
