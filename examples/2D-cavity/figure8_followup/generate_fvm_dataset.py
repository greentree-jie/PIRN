"""Generate all 101 cavity training fields solely from the finite-volume solver."""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from cavity_fvm import solve, to_vertex_grid

ROOT=Path(__file__).resolve().parent


def one_case(args):
    re,n,out=args
    u,v,p,diagnostics=solve(re,n)
    ug,vg,pg=to_vertex_grid(u,v,p,size=64)
    speed=re/10
    file=out/f'Re_{re:.1f}_N{n}.npz'
    np.savez_compressed(file,Re=re,N=n,u=ug*speed,v=vg*speed,p=pg*speed**2,
                        u_faces=u*speed,v_faces=v*speed,p_cells=p*speed**2,
                        diagnostics=np.array(str(diagnostics)))
    return re,str(file),diagnostics


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=64)
    ap.add_argument('--workers',type=int,default=min(4,os.cpu_count() or 1))
    args=ap.parse_args()
    out=ROOT/'fvm-reference';out.mkdir(exist_ok=True)
    reynolds=np.linspace(5,25,101)
    tasks=[(float(re),args.n,out) for re in reynolds]
    records={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(one_case,t) for t in tasks]
        for count,future in enumerate(as_completed(futures),1):
            re,path,diag=future.result();records[round(re,1)]=(path,diag)
            if count%10==0 or count==101:
                print(f'{count}/101 cases complete; latest Re={re:.1f}',flush=True)
    all_fields=[]
    for re in reynolds:
        z=np.load(records[round(float(re),1)][0])
        a=np.stack((z['u'],z['v'],z['p']))
        if a.shape!=(3,64,64) or not np.isfinite(a).all():
            raise RuntimeError(f'Invalid result at Re={re}')
        all_fields.append(a)
    fields=np.asarray(all_fields,dtype=np.float32)
    dataset=out/f'cavity_fvm_101_N{args.n}.npz'
    np.savez_compressed(dataset,Re=reynolds.astype(np.float32),fields=fields,
                        method='staggered finite volume; conservative flux; projection',
                        pressure_gauge='zero mean vertex grid each case',
                        geometry='L=0.1 m, nu=0.01 m2/s, U=Re/10 m/s')
    print(f'Saved {dataset} shape={fields.shape}; no OpenFOAM data used.',flush=True)


if __name__=='__main__':main()
