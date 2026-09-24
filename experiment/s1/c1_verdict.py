#!/usr/bin/env python3
"""C1 判定：Δ_illum − Δ_view（审稿指出 `same` 基线精确抵消，故等价于 AUC_illum − AUC_view）。
逐 seed 配对；同时按缺陷类型分层，检验 C1 是否只是 fracture 现象。"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from paired import wilcoxon_signed_rank_p as wp

def ci(ds, n=10000, seed=0):
    rng=np.random.default_rng(seed)
    bs=np.array([rng.choice(ds,len(ds),True).mean() for _ in range(n)])
    return float(np.quantile(bs,.025)), float(np.quantile(bs,.975))

def load(p):
    d=json.load(open(p)); A=defaultdict(dict)
    for r in d['rows']:
        A[(r['protocol'],r['k'],r['seed'],r.get('scope','all'))][r['domain']]=r['auroc']
    return d['meta'], A

def table(A, scope, label):
    print(f"\n{'='*96}\nC1 判定 [{label}]  scope={scope}   Δ = AUROC(illumination) − AUROC(view)\n{'='*96}")
    print(f"{'proto':8s} {'k':>2s} {'illum':>8s} {'view':>8s} {'Δ':>9s} {'sd':>8s} {'95%CI':>20s} {'p':>8s}  判定")
    print("-"*96)
    out=[]
    for proto in ('single','mixed'):
        for k in (1,2,4,8):
            ds=[];ai=[];av=[]
            for s in range(8):
                v=A.get((proto,k,s,scope))
                if v and 'illumination' in v and 'view' in v:
                    ds.append(v['illumination']-v['view']); ai.append(v['illumination']); av.append(v['view'])
            if len(ds)<3: continue
            ds=np.array(ds); lo,hi=ci(ds); p=wp(ds)
            verd = "光照更差" if hi<0 else ("视角更差" if lo>0 else "不可区分")
            print(f"{proto:8s} {k:>2d} {np.mean(ai):8.4f} {np.mean(av):8.4f} {ds.mean():+9.4f} "
                  f"{ds.std(ddof=1):8.4f} [{lo:+.4f},{hi:+.4f}] {p:8.4f}  {verd}")
            out.append(dict(protocol=proto,k=k,scope=scope,delta=float(ds.mean()),
                            lo=lo,hi=hi,p=p,verdict=verd))
    return out

if __name__=="__main__":
    meta,A=load(sys.argv[1])
    lab=f"masking={meta['masking']}"
    res=table(A,'all',lab)
    for d in ('fracture','groove','ablation','breakdown'):
        res+=table(A,d,lab)
    json.dump(res, open(sys.argv[2],'w'), indent=1, ensure_ascii=False)
    print(f"\n写入 {sys.argv[2]}")
