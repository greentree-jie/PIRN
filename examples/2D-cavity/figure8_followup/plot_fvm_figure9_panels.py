"""Recreate the manuscript Figure 9 b1/c1/d1 panel styling using FVM data.

Prediction/reference arrays come only from `fvm-results/Re_*_prediction_vs_FVM.npz`.
Panel layout and field colorbar ticks were read from the manuscript's embedded
Figure 9 images (image23, image25, image27). Error is the original-project
fraction abs((reference-prediction)/reference.max()), with NO percent scaling.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'fvm-results'
OUT=SOURCE/'figure9_original_style'

# (vmin, vmax, colorbar ticks) from the manuscript. For each field, both
# columns use the PIRN scale (the first tuple) for direct visual comparison.
SCALES={
  15: [((-0.2,1.4,np.arange(-.2,1.401,.2)),(-.18,1.26,np.arange(-.18,1.261,.18))),
       ((-.48,.48,np.arange(-.48,.481,.12)),(-.60,.48,np.arange(-.60,.481,.12))),
       ((-10,10,np.arange(-10,10.01,2.5)),(-10,10,np.arange(-10,10.01,2.5)))],
  20.1:[((-.25,1.75,np.arange(-.25,1.751,.25)),(-.25,1.75,np.arange(-.25,1.751,.25))),
        ((-.60,.60,np.arange(-.60,.601,.15)),(-.75,.60,np.arange(-.75,.601,.15))),
        ((-13.2,16.5,np.arange(-13.2,16.501,3.3)),(-12.8,16.0,np.arange(-12.8,16.001,3.2)))],
  30: [((-.4,2.8,np.arange(-.4,2.801,.4)),(-.36,2.52,np.arange(-.36,2.521,.36))),
       ((-.8,.8,np.arange(-.8,.801,.2)),(-1.25,.75,np.arange(-1.25,.751,.25))),
       ((-20,25,np.arange(-20,25.01,5)),(-20,25,np.arange(-20,25.01,5)))],
}

def color_format(re,row,col):
    if row==2:return '%.1f' if re!=30 else '%.0f'
    if re==15 and row==0 and col==1:return '%.2f'
    if re==20.1 and row in (0,1):return '%.2f'
    if re==30 and row==0 and col==1:return '%.2f'
    if re==30 and row==1 and col==1:return '%.2f'
    return '%.1f' if row==0 else '%.2f'

def axes_style(ax):
    ax.set_xlim(0,.1);ax.set_ylim(0,.1)
    ax.set_xticks([0,.05,.1]);ax.set_yticks([0,.05,.1])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both',labelsize=7,length=3,pad=1.5)
    ax.set_xlabel('x',fontsize=9,labelpad=1)
    ax.set_ylabel('y',fontsize=9,labelpad=7)
    ax.set_aspect('equal')
    for side in ('top','right'):
        ax.spines[side].set_visible(False)

def make_panel(re):
    z=np.load(SOURCE/f'Re_{re:g}_prediction_vs_FVM.npz')
    pred=z['prediction'];ref=z['reference']
    fig=plt.figure(figsize=(8.83,7.36),facecolor='white')
    fig.text(.5,.978,f'Re={re:g}',ha='center',va='center',fontsize=15)
    left=[.142,.442,.742];bottom=[.695,.385,.075]
    row_labels=[r'$u^{(1)}$',r'$u^{(2)}$',r'$p$']
    col_labels=['PIRN','Reference','error']
    max_errors={}
    for row in range(3):
        fig.text(.035,bottom[row]+.102,row_labels[row],ha='center',va='center',fontsize=13)
        for col in range(3):
            x=left[col];y=bottom[row]
            ax=fig.add_axes([x,y,.17,.204])
            cax=fig.add_axes([x+.182,y,.009,.204])
            axes_style(ax)
            ax.set_title(col_labels[col],fontsize=13,pad=7)
            if col<2:
                field=pred[row] if col==0 else ref[row]
                lo,hi,ticks=SCALES[re][row][0]
                im=ax.imshow(field,origin='lower',extent=(0,.1,0,.1),
                             cmap='jet',norm=Normalize(lo,hi,clip=True),interpolation='bilinear')
                ticks=np.where(np.abs(ticks)<1e-10,0,ticks)
                cb=fig.colorbar(im,cax=cax,ticks=ticks)
                cb.ax.yaxis.set_major_formatter(FormatStrFormatter(color_format(re,row,0)))
            else:
                denom=float(ref[row].max())
                if denom<=0:raise ValueError(f'Nonpositive reference maximum at Re={re:g}, row={row}')
                err=np.abs((ref[row]-pred[row])/denom)
                max_errors[row_labels[row]]=float(err.max())
                vmax=float(err.max())
                im=ax.imshow(err,origin='lower',extent=(0,.1,0,.1),
                             cmap='Blues',vmin=0,vmax=vmax,interpolation='bilinear')
                cb=fig.colorbar(im,cax=cax,ticks=np.linspace(0,vmax,10))
                cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
            cb.ax.tick_params(labelsize=7,length=3,pad=1.5)
    OUT.mkdir(parents=True,exist_ok=True)
    stem={15:'b1',20.1:'c1',30:'d1'}[re]
    fig.savefig(OUT/f'Figure9_{stem}_FVM_style.png',dpi=300,facecolor='white')
    fig.savefig(OUT/f'Figure9_{stem}_FVM_style.pdf',facecolor='white')
    plt.close(fig)
    return max_errors

def main():
    ims=[]
    for re in (15,20.1,30):
        errors=make_panel(re)
        print(f'Re={re:g}: raw fractional error maxima {errors}',flush=True)
        stem={15:'b1',20.1:'c1',30:'d1'}[re]
        ims.append(Image.open(OUT/f'Figure9_{stem}_FVM_style.png').convert('RGB'))
    w=max(im.width for im in ims)
    combined=Image.new('RGB',(w,sum(im.height for im in ims)),'white');y=0
    for im in ims:
        combined.paste(im,((w-im.width)//2,y));y+=im.height
    combined.save(OUT/'Figure9_b1_c1_d1_FVM_style.png',optimize=True)

if __name__=='__main__':main()
