"""Recreate Figure 9 b2/c2/d2 from the FVM-trained model and FVM references.

The original manuscript images show:
  b2: interpolation, Re=5.5,7.5,...,23.5
  c2: training point, Re=5,7,...,25
  d2: extrapolation, Re=26,27,...,35

Errors use the PIRN project's raw fractional convention:
    abs((reference - prediction) / reference.max())
No multiplication by 100 is performed. Missing FVM cases are solved on demand.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from cavity_fvm import solve, to_vertex_grid
from train_fvm_pirn import FvmPIRN, predict

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'fvm-results'/'figure9_original_style'
REF=ROOT/'fvm-reference'/'boxplot_cases'
CASES={
    'b2':('interpolation',np.arange(5.5,24,2,dtype=float)),
    'c2':('training point',np.arange(5,26,2,dtype=float)),
    'd2':('extrapolation',np.arange(26,36,1,dtype=float)),
}
COLORS={'u':'#1f77b4','v':'#ff7f0e','p':'#2ca02c'}


def solve_missing(re):
    u,v,p,diagnostics=solve(float(re),64)
    ug,vg,pg=to_vertex_grid(u,v,p,size=64)
    speed=re/10
    path=REF/f'Re_{re:g}_N64.npz'
    np.savez_compressed(path,Re=re,N=64,u=ug*speed,v=vg*speed,p=pg*speed**2,
                        diagnostics=np.array(str(diagnostics)))
    return float(re),str(path)


def ensure_references(workers):
    REF.mkdir(parents=True,exist_ok=True)
    wanted=np.r_[CASES['b2'][1],CASES['d2'][1]]
    missing=[]
    for re in wanted:
        if re==30 and (ROOT/'fvm-reference'/'Re_30_N64.npz').exists():continue
        if not (REF/f'Re_{re:g}_N64.npz').exists():missing.append(float(re))
    if missing:
        print(f'Computing {len(missing)} missing FVM reference cases...',flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(solve_missing,re) for re in missing]
            for count,future in enumerate(as_completed(futures),1):
                re,path=future.result()
                if count%5==0 or count==len(missing):
                    print(f'{count}/{len(missing)} complete; latest Re={re:g}',flush=True)


def reference(case,re,training_dataset):
    if case=='c2':
        re_all=training_dataset['Re'];idx=int(np.argmin(abs(re_all-re)))
        if abs(float(re_all[idx])-re)>1e-5:raise RuntimeError(f'Missing training Re={re:g}')
        a=training_dataset['fields'][idx].astype('float64')
    else:
        path=(ROOT/'fvm-reference'/'Re_30_N64.npz') if re==30 else REF/f'Re_{re:g}_N64.npz'
        z=np.load(path);a=np.stack([z[k] for k in ('u','v','p')])
    a[2]-=a[2].mean()
    if a.shape!=(3,64,64) or not np.isfinite(a).all():raise RuntimeError(f'Bad reference at Re={re:g}')
    return a


def errors_for_case(case,re_values,model,scale,dataset):
    result=np.empty((len(re_values),3,64,64),dtype='float32')
    for i,re in enumerate(re_values):
        ref=reference(case,float(re),dataset)
        pred=predict(model,float(re),scale)
        for j in range(3):
            denominator=float(ref[j].max())
            if denominator<=0:raise ValueError(f'Nonpositive reference maximum for Re={re:g}, field={j}')
            result[i,j]=np.abs((ref[j]-pred[j])/denominator)
    return result


def grouped_boxplot(case,title,re_values,errors):
    fig,ax=plt.subplots(figsize=(6.4,4.8),dpi=100)
    base=np.arange(len(re_values),dtype=float)
    offsets={'u':-.25,'v':0,'p':.25}
    handles=[]
    for j,name in enumerate(('u','v','p')):
        arrays=[errors[i,j].ravel() for i in range(len(re_values))]
        bp=ax.boxplot(arrays,positions=base+offsets[name],widths=.23,
                      patch_artist=True,showfliers=False,whis=1.5,
                      boxprops={'edgecolor':'#333333','linewidth':1.0},
                      whiskerprops={'color':'#333333','linewidth':1.0},
                      capprops={'color':'#333333','linewidth':1.0},
                      medianprops={'color':'#333333','linewidth':1.0})
        for box in bp['boxes']:box.set_facecolor(COLORS[name])
        handles.append(bp['boxes'][0])
    ax.set_xlim(-.5,len(re_values)-.5)
    ax.set_xticks(base)
    ax.set_xticklabels([f'{x:g}' for x in re_values])
    ax.set_xlabel('Re');ax.set_ylabel('error');ax.set_title(title)
    # The source plots keep all four spines and place the legend inside.
    for side in ('top','right','bottom','left'):ax.spines[side].set_visible(True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5,steps=[1,2,2.5,5,10]))
    ax.tick_params(axis='both',labelsize=9)
    ax.legend(handles,['u','v','p'],title='field',
              loc='upper left' if case=='d2' else 'upper center',
              frameon=True,fontsize=9,title_fontsize=9)
    fig.tight_layout()
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/f'Figure9_{case}_FVM_style.png',dpi=300)
    fig.savefig(OUT/f'Figure9_{case}_FVM_style.pdf')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args()
    ensure_references(args.workers)
    checkpoint=torch.load(ROOT/'fvm-results'/'fvm_pirn.pth',map_location='cpu')
    model=FvmPIRN(np.zeros((1,3,64,64),dtype='float32'))
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    scale=checkpoint['scale']
    dataset=np.load(ROOT/'fvm-reference'/'cavity_fvm_101_N64.npz')
    outputs={}
    for case,(title,re_values) in CASES.items():
        errors=errors_for_case(case,re_values,model,scale,dataset)
        grouped_boxplot(case,title,re_values,errors)
        outputs[f'{case}_Re']=re_values
        outputs[f'{case}_error']=errors
        print(f'{case}: Re={re_values.tolist()}, maxima={errors.max(axis=(0,2,3)).tolist()}',flush=True)
    np.savez_compressed(OUT/'Figure9_b2_c2_d2_error_samples.npz',**outputs)


if __name__=='__main__':main()
