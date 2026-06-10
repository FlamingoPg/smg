#!/usr/bin/env python3
"""
LLM-serving load-balancing simulator + algorithm bake-off.

Event-driven, iteration-level continuous-batching model with:
  - finite KV cache (admission control + preemption/recompute cliff)
  - concurrency cap (max_running)
  - linear iteration cost: t = base + per_tok * batch_tokens
  - chunked prefill under a per-iteration token budget
  - REALISTIC routing: the router sees STALE per-worker load (polled every
    poll_interval) plus an in-flight counter of dispatched-but-not-yet-polled
    requests -- this is what separates good algorithms from naive ones.

Experiment: each workload SHAPE is calibrated to a capacity (req/s) by
saturation; we then sweep utilization rho in {0.70,0.85,0.95}, run K seeds per
(shape,rho,policy), and report mean p99 latency. Lower = better; selection is
argmin (or power-of-d-choices). Pure stdlib. Deterministic given seeds.
"""
from __future__ import annotations
import heapq, math, random, statistics, time
from dataclasses import dataclass
from collections import deque, OrderedDict

PREFILL, DECODE, WAIT = 0, 1, 2
NOISE_CV = 0.0      # multiplicative lognormal noise on the polled load snapshot (0 = perfect info)
EWMA_ALPHA = 0.4    # smoothing for the EWMA-load variant (new sample weight)

@dataclass
class Req:
    rid: int; arrival: float; input_len: int; output_len: int
    state: int = WAIT
    prefill_remaining: int = 0
    generated: int = 0
    remaining_output: int = 0
    kv: int = 0
    preempted: int = 0
    session_id: int = -1           # for prefix-cache reuse (agentic multi-turn); -1 = no session
    prefix_len: int = 0            # cacheable history tokens (shared with prior turns of the session)
    pf_hit: int = 0                # prefix tokens served from cache (prefill skipped)
    t_admit: float | None = None   # first time it leaves the WAIT queue (queueing delay endpoint)
    t_first: float | None = None
    t_done: float | None = None
    worker: int = -1
    def context_len(self): return self.input_len + self.generated

class Worker:
    # iter_time = base + c_tok*(prefill+decode toks) + c_kv*(resident KV of decoders)
    #             + c_attn*(prefill attention work ~ chunk*context)
    def __init__(self, wid, kv_cap, max_running, base, c_tok, c_kv, c_attn, token_budget):
        self.wid=wid; self.kv_cap=kv_cap; self.max_running=max_running
        self.base=base; self.c_tok=c_tok; self.c_kv=c_kv; self.c_attn=c_attn
        self.token_budget=token_budget
        self.speed=1.0/c_tok
        self.running=[]; self.waiting=deque(); self.kv_used=0
        self.busy=False; self.cur_batch=None; self.completed=0
        self.snap={}; self.ewma=None; self.inflight_n=0; self.inflight_tok=0
        self.nrng=random.Random(7919+wid)
        self.pcache=OrderedDict()        # session_id -> cached prefix length (LRU); prefix cache
        self.pcache_budget=kv_cap        # tokens of prefix that can stay cached (shares KV when idle)
        self.snapshot()
    def run_tokens(self): return sum(r.kv for r in self.running)
    def wait_tokens(self): return sum(r.context_len() for r in self.waiting)
    def snapshot(self):
        raw=dict(run=len(self.running), wait=len(self.waiting),
                 run_tok=self.run_tokens(), wait_tok=self.wait_tokens(),
                 kv_used=self.kv_used, kv_cap=self.kv_cap,
                 max_running=self.max_running, speed=self.speed)
        if NOISE_CV>0:                                   # noisy load measurement (real systems)
            s=math.sqrt(math.log(1+NOISE_CV**2)); mu=-0.5*s*s
            for f in ('run','wait','run_tok','wait_tok','kv_used'):
                raw[f]=raw[f]*self.nrng.lognormvariate(mu,s)
        self.snap=raw
        if self.ewma is None: self.ewma=dict(raw)        # EWMA-smoothed snapshot
        else:
            for f in raw: self.ewma[f]=EWMA_ALPHA*raw[f]+(1-EWMA_ALPHA)*self.ewma[f]
        self.inflight_n=0; self.inflight_tok=0

ARR, ITER, POLL, END = 0,1,2,3

class Sim:
    def __init__(self, workers, workload, policy, poll_interval, duration):
        self.W=workers; self.policy=policy; self.poll=poll_interval
        self.duration=duration; self.t=0.0; self.q=[]; self._seq=0
        self.done=[]; self.preemptions=0
        for i,rec in enumerate(workload):
            at,inp,out=rec[0],rec[1],rec[2]
            sid=rec[3] if len(rec)>3 else -1
            plen=rec[4] if len(rec)>4 else 0
            r=Req(rid=i, arrival=at, input_len=inp, output_len=out, session_id=sid, prefix_len=plen)
            r.remaining_output=out
            self._push(at, ARR, r)
        for w in self.W: self._push(0.0, POLL, w)
    def _push(self,t,kind,p): heapq.heappush(self.q,(t,self._seq,kind,p)); self._seq+=1

    def start_iter(self, w, now):
        while len(w.running)<w.max_running and w.waiting:
            req=w.waiting[0]; need=req.context_len()
            if w.kv_used+need<=w.kv_cap:
                w.waiting.popleft(); req.state=PREFILL
                hit=0
                if req.session_id>=0 and req.preempted==0:        # prefix-cache hit (first admit)
                    hit=min(req.prefix_len, w.pcache.get(req.session_id,0))
                req.prefill_remaining=max(1, need-hit); req.pf_hit=hit  # skip cached prefix prefill
                if req.t_admit is None: req.t_admit=now   # queueing delay = t_admit - arrival
                req.kv=need; w.kv_used+=need; w.running.append(req)
            else: break
        if not w.running: w.busy=False; return
        budget=w.token_budget; batch=[]
        P=0; D=0; kv_decode=0.0; pre_attn=0.0
        for r in w.running:
            if r.state==DECODE:
                batch.append((r,DECODE,1)); D+=1; kv_decode+=r.kv      # decode attends to full KV
        rem=budget-D
        for r in w.running:
            if r.state==PREFILL and rem>0:
                chunk=min(r.prefill_remaining,rem)
                ctx0=r.context_len()-r.prefill_remaining               # tokens already prefilled
                pre_attn += chunk*(ctx0 + chunk/2.0)                   # chunk attends to prior context
                batch.append((r,PREFILL,chunk)); P+=chunk; rem-=chunk
        iter_time = w.base + w.c_tok*(P+D) + w.c_kv*kv_decode + w.c_attn*pre_attn
        w.cur_batch=batch; w.busy=True
        self._push(now + iter_time, ITER, w)

    def finish_iter(self, w, now):
        comp=[]
        for (r,kind,chunk) in w.cur_batch:
            if kind==PREFILL:
                r.prefill_remaining-=chunk
                if r.prefill_remaining==0:
                    r.state=DECODE
                    if r.t_first is None: r.t_first=now
                    r.generated+=1; r.kv+=1; w.kv_used+=1; r.remaining_output-=1
                    if r.remaining_output==0: comp.append(r)
            else:
                r.generated+=1; r.kv+=1; w.kv_used+=1; r.remaining_output-=1
                if r.remaining_output==0: comp.append(r)
        for r in comp:
            w.kv_used-=r.kv; r.kv=0; w.running.remove(r); r.t_done=now
            w.completed+=1; self.done.append(r)
            if r.session_id>=0:                              # cache this session's full context (LRU)
                w.pcache[r.session_id]=r.input_len; w.pcache.move_to_end(r.session_id)
                while sum(w.pcache.values())>w.pcache_budget and len(w.pcache)>1:
                    w.pcache.popitem(last=False)
        while w.kv_used>w.kv_cap and w.running:
            v=w.running.pop(); w.kv_used-=v.kv; v.kv=0; v.state=WAIT
            v.prefill_remaining=v.context_len(); v.preempted+=1
            self.preemptions+=1; w.waiting.appendleft(v)
        self.start_iter(w, now)

    def route(self, r, now):
        w=self.policy.select(self.W, r); r.worker=w.wid
        w.waiting.append(r); w.inflight_n+=1; w.inflight_tok+=r.input_len
        if not w.busy: self.start_iter(w, now)

    def run(self):
        self._push(self.duration, END, None)
        while self.q:
            t,_,kind,p=heapq.heappop(self.q); self.t=t
            if kind==END: break
            if kind==ARR: self.route(p,t)
            elif kind==ITER: self.finish_iter(p,t)
            elif kind==POLL:
                p.snapshot()
                if t+self.poll<self.duration: self._push(t+self.poll,POLL,p)
        return self.metrics()

    def metrics(self):
        done=[r for r in self.done if r.t_first is not None]
        qd=sorted(r.t_admit-r.arrival for r in done if r.t_admit is not None)  # PURE queueing delay
        ttft=sorted(r.t_first-r.arrival for r in done)
        e2e=sorted(r.t_done-r.arrival for r in done if r.t_done is not None)
        tpot=[(r.t_done-r.t_first)/max(1,r.generated-1) for r in done if r.t_done and r.generated>1]
        per=[w.completed for w in self.W]; mc=statistics.mean(per) if per else 0
        cov=(statistics.pstdev(per)/mc) if mc>0 else 0.0
        def pct(xs,p): return float('nan') if not xs else xs[min(len(xs)-1,int(p/100*len(xs)))]
        sess=[r for r in done if r.session_id>=0]
        pf=sum(r.pf_hit for r in sess)/max(1,sum(r.input_len for r in sess))
        return dict(n=len(self.done), pf_rate=pf, qd_p50=pct(qd,50), qd_p99=pct(qd,99),
                    ttft_p50=pct(ttft,50), ttft_p99=pct(ttft,99),
                    e2e_p50=pct(e2e,50), e2e_p99=pct(e2e,99),
                    tpot=statistics.mean(tpot) if tpot else float('nan'),
                    thru=sum(r.generated for r in done)/self.duration,
                    preempt=self.preemptions, cov=cov)

# ---------------------------------------------------------------- policies
def _k(s): return min(0.999, s['kv_used']/max(1,s['kv_cap']))
def _bar(s, lam=2.0):
    k=_k(s); return lam*k/(1.0-k)

class Policy:
    def __init__(self,name,fn): self.name=name; self._fn=fn
    def select(self,ws,req=None): return self._fn(ws,req)

def make_policies(seed=0):
    rng=random.Random(seed); rr={'i':0}; P={}
    def argmin(score):
        def sel(ws,req=None):
            best=None; bv=None
            for w in ws:
                v=score(w)
                if bv is None or v<bv: bv=v; best=w
            return best
        return sel
    def powd(score,d=2):
        def sel(ws,req=None):
            if len(ws)<=d: return min(ws,key=score)
            cand=rng.sample(ws,d); return min(cand,key=score)
        return sel
    # --- baselines
    P['random']=Policy('random', lambda ws,req=None: rng.choice(ws))
    def _rr(ws,req=None):
        w=ws[rr['i']%len(ws)]; rr['i']+=1; return w
    P['round_robin']=Policy('round_robin', _rr)
    # --- request-count family
    P['jsq']=Policy('jsq(stale,no-if)', argmin(lambda w: w.snap['run']+w.snap['wait']))
    P['jsq+if']=Policy('jsq+if', argmin(lambda w: w.snap['run']+w.snap['wait']+w.inflight_n))
    P['pow2+if']=Policy('pow2+if', powd(lambda w: w.snap['run']+w.snap['wait']+w.inflight_n,2))
    P['wjsq+if']=Policy('wjsq+if(cap-weighted)',
        argmin(lambda w: (w.snap['run']+w.snap['wait']+w.inflight_n)/w.snap['speed']*1e5))
    def _jiq(ws,req=None):
        idle=[w for w in ws if w.snap['run']+w.inflight_n==0]
        if idle: return rng.choice(idle)
        c=rng.sample(ws,2) if len(ws)>2 else ws
        return min(c,key=lambda w: w.snap['run']+w.snap['wait']+w.inflight_n)
    P['jiq+if']=Policy('jiq+if(idle-first)', _jiq)
    # --- KV / utilization family
    P['least_kv']=Policy('least_kv', argmin(lambda w: _k(w.snap)+w.inflight_tok/max(1,w.snap['kv_cap'])))
    P['lutil+if']=Policy('lutil+if(bottleneck)',
        argmin(lambda w: max((w.snap['run']+w.inflight_n)/max(1,w.snap['max_running']),
                             (w.snap['kv_used']+w.inflight_tok)/max(1,w.snap['kv_cap']))))
    # --- HALO family
    P['halo+if']=Policy('halo+if', argmin(lambda w: w.snap['run']+w.inflight_n+_bar(w.snap)))
    P['halo-tok']=Policy('halo-tok+if',
        argmin(lambda w: (w.snap['wait_tok']+w.inflight_tok)/2000.0 + _bar(w.snap)))
    # --- token / expected-wait family (PROPOSED)
    P['lqt+if']=Policy('lqt+if(least q-toks)', argmin(lambda w: w.snap['wait_tok']+w.inflight_tok))
    def ew_score(w):
        return (w.snap['wait_tok']+w.inflight_tok)/w.snap['speed'] + _bar(w.snap)
    def ewf_score(w):  # running-aware
        return (w.snap['wait_tok']+w.inflight_tok+0.4*w.snap['run_tok'])/w.snap['speed'] + _bar(w.snap)
    P['EW']=Policy('EW(units-bug)', argmin(ew_score))
    P['EW-full']=Policy('EW-full(run-aware)', argmin(ewf_score))
    P['EW-pow2']=Policy('EW-pow2', powd(ew_score,2))
    P['EW-d3']=Policy('EW-d3', powd(ew_score,3))
    # UNIT-CONSISTENT proposed score: both terms in SECONDS (expected queue wait + KV-pressure time)
    LAMBDA_T=0.15  # seconds: time-cost weight of KV pressure (commensurate with the queue-time term)
    def ewq_score(w):
        s=w.snap; k=_k(s)
        return (s['wait_tok']+w.inflight_tok)/s['speed'] + LAMBDA_T*k/(1.0-k)
    P['EWQ']=Policy('EWQ(time-consistent)', argmin(ewq_score))
    P['EWQ-pow2']=Policy('EWQ-pow2', powd(ewq_score,2))
    # --- ablations of EWQ (attribute the win to each term) ---
    def _kk(s): return min(0.999, s['kv_used']/max(1,s['kv_cap']))
    P['EWQ-noBar']=Policy('EWQ-noBar(no KV)', argmin(lambda w:(w.snap['wait_tok']+w.inflight_tok)/w.snap['speed']))
    P['EWQ-noIF'] =Policy('EWQ-noIF(no inflight)',
        argmin(lambda w: w.snap['wait_tok']/w.snap['speed'] + LAMBDA_T*_kk(w.snap)/(1-_kk(w.snap))))
    P['EWQ-noSpeed']=Policy('EWQ-noSpeed(no /mu)',
        argmin(lambda w:(w.snap['wait_tok']+w.inflight_tok)/30000.0 + LAMBDA_T*_kk(w.snap)/(1-_kk(w.snap))))
    P['EWQ-count']=Policy('EWQ-count(reqs not toks)',
        argmin(lambda w:(w.snap['wait']+w.inflight_n)*0.05 + LAMBDA_T*_kk(w.snap)/(1-_kk(w.snap))))
    # --- EWMA-smoothed load (helps under noisy measurements) ---
    def ewma_score(w):
        s=w.ewma or w.snap; k=min(0.999, s['kv_used']/max(1,s['kv_cap']))
        return (s['wait_tok']+w.inflight_tok)/s['speed'] + LAMBDA_T*k/(1-k)
    P['EWQ-ewma']=Policy('EWQ-ewma(smoothed)', argmin(ewma_score))
    # --- oracle reference (cheats: live state, no staleness)
    P['jsq_true']=Policy('jsq_true(ORACLE)', argmin(lambda w: len(w.running)+len(w.waiting)))
    # --- prefix-cache-aware (for agentic multi-turn: route a session to where its prefix is cached) ---
    prefix_loc={}
    def _byid(ws): return {w.wid:w for w in ws}
    def cache_aff(ws,req=None):
        bi=_byid(ws)
        if req is not None and req.session_id>=0 and prefix_loc.get(req.session_id) in bi:
            w=bi[prefix_loc[req.session_id]]          # sticky to cached worker (ignores load)
        else:
            w=min(ws,key=ewq_score)
        if req is not None and req.session_id>=0: prefix_loc[req.session_id]=w.wid
        return w
    def cache_gate(ws,req=None,gate=1.6):
        sc={w.wid:ewq_score(w) for w in ws}; best=min(sc.values()); bi=_byid(ws); w=None
        if req is not None and req.session_id>=0 and prefix_loc.get(req.session_id) in bi:
            cand=bi[prefix_loc[req.session_id]]       # cache hit IF not too overloaded (HALO-style gate)
            if sc[cand.wid]<=gate*best+1e-9: w=cand
        if w is None: w=min(ws,key=lambda x:sc[x.wid])
        if req is not None and req.session_id>=0: prefix_loc[req.session_id]=w.wid
        return w
    P['cache_aff']=Policy('cache_affinity(sticky)', cache_aff)
    P['cache_gate']=Policy('cache+EWQ-gate(HALO-style)', cache_gate)
    return P

# ---------------------------------------------------------------- workloads
def lognormal(rng, mean, cv=0.7):
    sigma=math.sqrt(math.log(1+cv*cv)); mu=math.log(max(1.0,mean))-0.5*sigma*sigma
    return max(1,int(rng.lognormvariate(mu,sigma)))

SIZE_PROFILES={'tiny':(80,40),'chat':(400,200),'code':(1500,600),
               'longchat':(2200,800),'rag':(6000,300),'huge':(16000,500),
               # agentic: long, diverse inputs (tool outputs / growing history), variable output
               'toolcall':(1200,80),'agentic':(12000,400),'agentic_huge':(32000,700)}

def _draw(rng, prof, cv, max_ctx):
    mi,mo=SIZE_PROFILES[prof]
    inp=min(max_ctx-1, lognormal(rng,mi,cv)); o=lognormal(rng,mo,cv)
    return inp, max(1,min(o,max_ctx-inp-1))

def gen_workload(rng, rate, duration, profile_mix, bursty=False, burst_k=20,
                 max_ctx=44000, size_cv=0.7, rate_profile='flat'):
    names=[p for p,_ in profile_mix]; wts=[w for _,w in profile_mix]; out=[]
    def rate_at(t):
        if rate_profile=='diurnal': return rate*(1.0+0.85*math.sin(2*math.pi*t/(duration/2.0)))
        if rate_profile=='rampup':  return rate*(0.2+1.6*t/duration)
        return rate
    peak=rate*(1.9 if rate_profile in ('diurnal','rampup') else 1.0)
    t=0.0
    if bursty:
        gr=rate/burst_k
        while True:
            t+=rng.expovariate(gr)
            if t>=duration: break
            for _ in range(burst_k):
                inp,o=_draw(rng, rng.choices(names,wts)[0], size_cv, max_ctx); out.append((t,inp,o))
    else:
        while True:
            t+=rng.expovariate(peak)
            if t>=duration: break
            if rate_profile!='flat' and rng.random()>rate_at(t)/peak: continue
            inp,o=_draw(rng, rng.choices(names,wts)[0], size_cv, max_ctx); out.append((t,inp,o))
    out.sort(); return out

def gen_agentic(rng, rate, duration, n_sessions=150, new_mean=1500, out_mean=350, sys_prompt=1500, max_ctx=40000):
    """Multi-turn agentic sessions: each turn's input = shared history (cacheable prefix) + new tool/query
    tokens; history grows each turn. Fewer sessions => more prefix reuse. Returns 5-tuples."""
    ctx={}; out=[]; t=0.0
    while True:
        t+=rng.expovariate(rate)
        if t>=duration: break
        sid=rng.randrange(n_sessions)
        c=ctx.get(sid, sys_prompt)
        new=lognormal(rng, new_mean, 1.2)
        inp=min(max_ctx-1, c+new); plen=min(c, inp-1)
        o=max(1, min(lognormal(rng,out_mean,0.8), max_ctx-inp-1))
        out.append((t,inp,o,sid,plen)); ctx[sid]=min(max_ctx-1, inp+o)
    out.sort(); return out

# ---------------------------------------------------------------- worker fleets
# default per-worker cost coefficients (calibrated for realistic TTFT/TPOT magnitudes; see validate())
DEF=dict(base=0.006, c_tok=3e-5, c_kv=2e-7, c_attn=1e-8, budget=8192)
def homog_workers(n, kv=120000, maxr=256, **kw):
    p={**DEF,**kw}
    return lambda: [Worker(i,kv,maxr,p['base'],p['c_tok'],p['c_kv'],p['c_attn'],p['budget']) for i in range(n)]
def hetero_workers(n, specs):
    # spec: {'mult': scales all cost coeffs (>1 = slower), 'kv': capacity}
    def build():
        ws=[]
        for i in range(n):
            s=specs[i%len(specs)]; m=s.get('mult',1.0)
            ws.append(Worker(i, s.get('kv',120000), s.get('maxr',256),
                             DEF['base']*m, DEF['c_tok']*m, DEF['c_kv']*m, DEF['c_attn']*m, DEF['budget']))
        return ws
    return build

# ---------------------------------------------------------------- SHAPES (diverse)
SHAPES={
 'chat':    dict(workers=homog_workers(8), mix=[('chat',1.0)], poll=0.5,
                 note='8 homog workers, short chat (sanity: JSQ family should win)'),
 'mixed':   dict(workers=homog_workers(8), mix=[('chat',0.8),('rag',0.2)], poll=0.5,
                 note='8 homog, mixed sizes 80%chat/20%rag (token-aware matters)'),
 'bigmix':  dict(workers=homog_workers(16), mix=[('tiny',0.3),('chat',0.4),('code',0.15),('rag',0.1),('huge',0.05)],
                 poll=0.5, size_cv=0.8, note='16 homog, VERY diverse tiny..huge'),
 'heavytail':dict(workers=homog_workers(16), mix=[('chat',0.85),('huge',0.15)], poll=0.5, size_cv=1.4,
                 note='16 homog, heavy-tailed sizes cv=1.4 (long-context outliers)'),
 'bursty':  dict(workers=homog_workers(8), mix=[('chat',0.8),('rag',0.2)], bursty=True, burst_k=25, poll=1.0,
                 note='8 homog, INCAST bursts k=25 + 1s poll (herding stress)'),
 'diurnal': dict(workers=homog_workers(8), mix=[('chat',0.8),('rag',0.2)], poll=0.5, rate_profile='diurnal',
                 note='8 homog, time-varying diurnal load + mixed sizes'),
 'hetero':  dict(workers=hetero_workers(8,[{'mult':0.8,'kv':160000},{'mult':1.8,'kv':80000}]),
                 mix=[('chat',0.7),('rag',0.3)], poll=0.5,
                 note='4 fast/big + 4 slow/small workers, mixed sizes'),
 'hetero3': dict(workers=hetero_workers(9,[{'mult':0.7,'kv':200000},{'mult':1.2,'kv':120000},
                                           {'mult':2.2,'kv':60000}]),
                 mix=[('chat',0.7),('rag',0.3)], poll=0.5, note='9 workers, 3 speed/KV tiers'),
 'largefleet':dict(workers=homog_workers(16), mix=[('chat',0.8),('rag',0.2)], poll=0.5,
                 note='16 homog workers, mixed sizes (fleet scale)'),
 'agentic':  dict(workers=homog_workers(16, kv=180000),
                 mix=[('toolcall',0.5),('agentic',0.35),('agentic_huge',0.15)], poll=0.5, size_cv=1.0,
                 note='16 workers, AGENTIC: long+diverse inputs (toolcalls + 12k + 32k context)'),
 'agentic_burst':dict(workers=homog_workers(16, kv=180000),
                 mix=[('toolcall',0.5),('agentic',0.35),('agentic_huge',0.15)], bursty=True, burst_k=12,
                 poll=1.0, size_cv=1.0, note='agentic + session-fanout bursts'),
}

def calibrate_capacity(cfg, dur=35.0, slo=0.5):
    """Capacity (req/s) = QUEUEING KNEE: max arrival rate where round_robin keeps queue-delay.p99
    (t_admit-arrival) < slo. NOT TTFT: for large inputs TTFT.p99 is floored by prefill time
    (16k tokens => 1.8s) which masks the queueing knee. Queue-delay is what routing controls."""
    def qd99(lam):
        rng=random.Random(7)
        wl=gen_workload(rng, lam, dur, cfg['mix'], bursty=cfg.get('bursty',False),
                        burst_k=cfg.get('burst_k',20), size_cv=cfg.get('size_cv',0.7),
                        rate_profile=cfg.get('rate_profile','flat'))
        workers=cfg['workers']()
        return Sim(workers, wl, make_policies(seed=999)['round_robin'], cfg['poll'], dur).run()['qd_p99']
    lo,hi=1.0,1.0
    for _ in range(16):                       # exp-grow until queue-delay.p99 exceeds slo
        if qd99(hi)<slo: lo=hi; hi*=1.5
        else: break
    for _ in range(8):                        # bisection on the knee
        mid=(lo+hi)/2
        if qd99(mid)<slo: lo=mid
        else: hi=mid
    return lo

# ---------------------------------------------------------------- experiment
CONTENDERS=['round_robin','jsq','jsq+if','pow2+if','wjsq+if','jiq+if','least_kv','lutil+if',
            'halo+if','halo-tok','lqt+if','EW','EW-pow2','EWQ','EWQ-pow2','jsq_true']
RHOS=[0.85,0.95,1.05]; SEEDS=[1,2,3,4,5]; MEAS_DUR=35.0   # 1.05 = just past round_robin's knee

def one_run(cfg, lam, pname, seed):
    rng=random.Random(1000+seed)
    wl=gen_workload(rng, lam, MEAS_DUR, cfg['mix'], bursty=cfg.get('bursty',False),
                    burst_k=cfg.get('burst_k',20), size_cv=cfg.get('size_cv',0.7),
                    rate_profile=cfg.get('rate_profile','flat'))
    workers=cfg['workers']()
    sim=Sim(workers, wl, make_policies(seed=999)[pname], cfg['poll'], MEAS_DUR)
    return sim.run()

def run_experiment():
    out={}
    for shape,cfg in SHAPES.items():
        C=calibrate_capacity(cfg)
        rows={}
        for p in CONTENDERS:
            row={}
            for rho in RHOS:
                lam=rho*C; qd=[]; ttft=[]; e2e=[]; pre=[]
                for s in SEEDS:
                    m=one_run(cfg,lam,p,s)
                    qd.append(m['qd_p99']); ttft.append(m['ttft_p99']); e2e.append(m['e2e_p99']); pre.append(m['preempt'])
                row[rho]=dict(qd=statistics.mean(qd), sd=(statistics.pstdev(qd) if len(qd)>1 else 0.0),
                              ttft=statistics.mean(ttft), e2e=statistics.mean(e2e), pre=statistics.mean(pre))
            rows[p]=row
        out[shape]={'C':C,'rows':rows}
        print(f'[calibrated {shape}: ~{C:.0f} req/s]', flush=True)
    return out

def rname(p):
    pol=make_policies().get(p); return pol.name if pol else p

def print_experiment(out):
    wins={}; sig_wins={}
    for shape,cfg in SHAPES.items():
        d=out[shape]
        print('\n'+'='*118)
        print(f'SHAPE: {shape}   capacity~{d["C"]:.0f} req/s   {cfg["note"]}')
        print('-'*118)
        print(f'{"policy":26}'+''.join(f'{("rho=%.2f QD.p99±sd / TTFT.p99"%r):>32}' for r in RHOS)+f'{"pre@.95":>8}')
        best={r:min(d['rows'][p][r]['qd'] for p in CONTENDERS if p!='jsq_true') for r in RHOS}
        second={}
        for r in RHOS:
            vals=sorted(d['rows'][p][r]['qd'] for p in CONTENDERS if p!='jsq_true')
            second[r]=vals[1] if len(vals)>1 else vals[0]
        for p in CONTENDERS:
            line=f'{rname(p):26}'
            for r in RHOS:
                x=d['rows'][p][r]
                mk=' '
                if p!='jsq_true' and abs(x['qd']-best[r])<1e-9:
                    mk='*'; wins[p]=wins.get(p,0)+1
                    if x['qd']+x['sd'] < second[r]-1e-9: sig_wins[p]=sig_wins.get(p,0)+1
                line+=f'{x["qd"]*1000:8.0f}±{x["sd"]*1000:<4.0f}/{x["ttft"]*1000:<8.0f}{mk}'
            line+=f'{d["rows"][p][RHOS[-1]]["pre"]:8.0f}'
            print(line)
    print('\n'+'='*118)
    print('OVERALL WINS by queue-delay.p99 (shape×rho where policy is best; oracle excluded):')
    for p,c in sorted(wins.items(), key=lambda x:-x[1]):
        print(f'   {rname(p):26} wins={c:2d}   significant(lead>1sd)={sig_wins.get(p,0)}')
    print('(cell = QUEUE-DELAY.p99 ± seed-stdev / TTFT.p99, ms, mean of %d seeds; pre@.95 = preemptions at rho=0.95)'%len(SEEDS))
    print('NOTE: rank on queue-delay.p99 = the pure routing-controlled metric (t_admit-arrival),')
    print('      excluding intrinsic prefill+decode time. TTFT shown for context.')

# ---------------------------------------------------------------- validation
def _one_req_prefill(inp):
    w=homog_workers(1, kv=600000)()
    sim=Sim(w, [(0.0, inp, 2)], make_policies()['round_robin'], 100.0, 60.0); sim.run()
    r=sim.done[0]; return r.t_first - r.arrival
def _decode_tpot(B, L):                      # steady decode iter time (model formula)
    d=DEF; return d['base'] + d['c_tok']*B + d['c_kv']*B*L

def validate():
    print('='*72); print('WORKER-MODEL VALIDATION (must hold before any ranking is trusted)'); print('='*72)
    ok=True
    print('\n[V1] single-request prefill time vs input length (no contention) -- expect super-linear:')
    Ls=[256,1024,4096,16000]; p=[_one_req_prefill(L) for L in Ls]
    for L,t in zip(Ls,p): print(f'     input={L:6d}  prefill={t*1000:8.1f} ms   ({L/t:8.0f} tok/s effective)')
    mono=all(p[i]<p[i+1] for i in range(len(p)-1)); ratio=p[-1]/p[0]; lin=Ls[-1]/Ls[0]
    print(f'     monotonic={mono}  ratio(16000/256)={ratio:.0f} vs linear={lin:.0f}  super-linear={ratio>lin}')
    ok &= mono and ratio>lin
    print('\n[V2] decode TPOT vs batch size (ctx=512) -- expect rising (batch contention):')
    bs=[1,8,32,128,256]; tb=[_decode_tpot(B,512) for B in bs]
    for B,t in zip(bs,tb): print(f'     batch={B:4d}  TPOT={t*1000:6.1f} ms')
    ok &= all(tb[i]<tb[i+1] for i in range(len(tb)-1))
    print('\n[V3] decode TPOT vs context length (batch=32) -- expect rising (KV attention):')
    cs=[256,1024,4096,12000]; tc=[_decode_tpot(32,L) for L in cs]
    for L,t in zip(cs,tc): print(f'     ctx={L:6d}  TPOT={t*1000:6.1f} ms')
    ok &= all(tc[i]<tc[i+1] for i in range(len(tc)-1))
    print('\n[V4] queue-delay.p99 vs utilization (chat, round_robin) -- expect hockey-stick:')
    cfg=SHAPES['chat']; C=calibrate_capacity(cfg); print(f'     calibrated knee ~ {C:.0f} req/s')
    rr=[]
    for rho in [0.5,0.8,0.95,1.05,1.2]:
        m=one_run(cfg, rho*C, 'round_robin', 1); rr.append(m['qd_p99'])
        print(f'     rho={rho:.2f}  QD.p99={m["qd_p99"]*1000:8.0f} ms  TTFT.p99={m["ttft_p99"]*1000:8.0f}  e2e.p99={m["e2e_p99"]*1000:8.0f}  preempt={m["preempt"]}')
    hockey=rr[-1]>4*max(1e-3,rr[0]); print(f'     hockey-stick(QD.p99 .5->1.2 >4x)={hockey}'); ok &= hockey
    print('\n[V5] preemptions -- expect 0 at low load, >0 under overload (KV fills past the knee):')
    lo=one_run(SHAPES['chat'], 0.4*C, 'round_robin',1)['preempt']
    hi=one_run(SHAPES['chat'], 1.3*C, 'round_robin',1)['preempt']
    print(f'     chat@0.4xknee={lo}   chat@1.3xknee={hi}'); ok &= (lo==0 and hi>0)
    print('\n'+('>>> VALIDATION PASSED' if ok else '>>> VALIDATION FAILED -- fix model before trusting results'))
    return ok

# ---------------------------------------------------------------- extra tests
ABL_MIX={'chat':[('chat',1.0)],
         'mixed':[('chat',0.8),('rag',0.2)],
         'agentic':[('toolcall',0.5),('agentic',0.35),('agentic_huge',0.15)]}
def _cfg(mix, workerkw=None, cv=0.7, poll=0.5):
    return dict(workers=homog_workers(16, kv=180000, **(workerkw or {})), mix=mix, poll=poll, size_cv=cv)
def _mean_qd(cfg, lam, p, seeds=(1,2,3)):
    return statistics.mean(one_run(cfg,lam,p,s)['qd_p99'] for s in seeds)

def ablate():
    pols=['EWQ','EWQ-noBar','EWQ-noIF','EWQ-noSpeed','EWQ-count','halo-tok','least_kv','jsq+if','round_robin']
    print('ABLATION  (queue-delay.p99 ms, rho=0.95, mean 3 seeds) -- which EWQ term carries the win?')
    for name,mix in ABL_MIX.items():
        cfg=_cfg(mix, cv=1.0 if name=='agentic' else 0.7); C=calibrate_capacity(cfg); lam=0.95*C
        res={p:_mean_qd(cfg,lam,p) for p in pols}
        print(f'\n  {name} (knee~{C:.1f} req/s):')
        for p in sorted(res,key=res.get): print(f'     {rname(p):26} {res[p]*1000:9.1f}')

def robust():
    pert={'baseline':{}, 'c_kv x2':{'c_kv':DEF["c_kv"]*2}, 'c_attn x2':{'c_attn':DEF["c_attn"]*2},
          'base x2':{'base':DEF["base"]*2}, 'poll 4x':{}}
    pols=['round_robin','jsq+if','pow2+if','least_kv','lutil+if','halo-tok','EWQ','EWQ-pow2']
    print('ROBUSTNESS (queue-delay.p99 ms, rho=0.95, mean 3 seeds) -- does EWQ stay best under perturbation?')
    for name,mix in ABL_MIX.items():
        print(f'\n  {name}:')
        for pn,ov in pert.items():
            poll=2.0 if pn=='poll 4x' else 0.5
            cfg=_cfg(mix, ov, cv=1.0 if name=='agentic' else 0.7, poll=poll)
            C=calibrate_capacity(cfg); lam=0.95*C
            res={p:_mean_qd(cfg,lam,p) for p in pols}
            win=min(res,key=res.get)
            print(f'     {pn:9} knee~{C:5.1f}  WINNER={rname(win).split("(")[0]:14} '
                  f'EWQ={res["EWQ"]*1000:7.0f}  best={res[win]*1000:7.0f}  rr={res["round_robin"]*1000:7.0f}')

def noise_test():
    global NOISE_CV
    pols=['jsq+if','least_kv','halo-tok','EWQ','EWQ-pow2','EWQ-ewma']
    print('NOISE TEST (queue-delay.p99 ms, rho=0.95, mean 3 seeds) -- does EWMA help under noisy load info?')
    for nc in [0.0,0.4,0.8]:
        NOISE_CV=nc
        print(f'\n  NOISE_CV={nc}:')
        for name in ['chat','mixed']:
            cfg=_cfg(ABL_MIX[name]); C=calibrate_capacity(cfg); lam=0.95*C
            res={p:_mean_qd(cfg,lam,p) for p in pols}
            print(f'     {name}: '+'  '.join(f'{rname(p).split("(")[0]}={res[p]*1000:.0f}' for p in pols))
    NOISE_CV=0.0

def prefix():
    wf=homog_workers(16, kv=180000); dur=40.0
    pols=['round_robin','jsq+if','EWQ','EWQ-pow2','cache_aff','cache_gate']
    print('PREFIX-CACHE / AGENTIC SESSIONS (mean 3 seeds; rho=0.9 of round_robin knee)')
    print('fewer sessions = MORE prefix reuse (cache-affinity should matter more)')
    def runp(lam,p,seed,ns):
        wl=gen_agentic(random.Random(1000+seed), lam, dur, n_sessions=ns)
        return Sim(wf(), wl, make_policies(seed=999)[p], 0.5, dur).run()
    for ns in [40,150,600]:
        lo,hi=0.5,0.5
        for _ in range(16):
            if runp(hi,'round_robin',7,ns)['qd_p99']<0.5: lo=hi; hi*=1.5
            else: break
        for _ in range(7):
            mid=(lo+hi)/2
            if runp(mid,'round_robin',7,ns)['qd_p99']<0.5: lo=mid
            else: hi=mid
        C=lo; lam=0.9*C
        print(f'\n  n_sessions={ns}  knee~{C:.1f} req/s   (QD.p99 / TTFT.p99 ms / cache-hit%):')
        for p in pols:
            ms=[runp(lam,p,s,ns) for s in (1,2,3)]
            qd=statistics.mean(m['qd_p99'] for m in ms)*1000
            tt=statistics.mean(m['ttft_p99'] for m in ms)*1000
            ch=statistics.mean(m['pf_rate'] for m in ms)*100
            print(f'     {rname(p):28} QD={qd:8.0f}  TTFT={tt:9.0f}  cache_hit={ch:5.1f}%')

if __name__=='__main__':
    import sys
    cmd=sys.argv[1] if len(sys.argv)>1 else 'run'
    t0=time.time()
    if cmd=='validate': validate()
    elif cmd=='ablate': ablate()
    elif cmd=='robust': robust()
    elif cmd=='noise': noise_test()
    elif cmd=='prefix': prefix()
    else: out=run_experiment(); print_experiment(out)
    print(f'\n[done in {time.time()-t0:.0f}s]')
