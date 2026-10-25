"""Train a three-branch residual model exclusively on the 101 FVM fields.

Run after generate_fvm_dataset.py. Re=20.1 and 30 references are independent
additional FVM solves at the same N=64 grid, never included in training.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'fvm-results'


def draw_schematic(out):
    fig,ax=plt.subplots(figsize=(14,2.2),constrained_layout=True)
    ax.plot([0,1,1,0,0],[0,0,1,1,0],color='black',lw=2)
    ax.arrow(.15,1.08,.65,0,width=.008,head_width=.06,head_length=.08,
             color='#b22222',length_includes_head=True)
    ax.text(.5,1.18,'Moving lid: $u=U$, $v=0$',ha='center')
    ax.text(.5,-.13,'No slip: $u=v=0$',ha='center')
    ax.text(-.15,.5,'No slip',ha='right',rotation=90,va='center')
    ax.text(1.15,.5,'No slip',ha='left',rotation=90,va='center')
    ax.text(1.75,.64,'$L=0.1$ m, $\\nu=0.01$ m$^2$/s',fontsize=13)
    ax.text(1.75,.35,'$Re=UL/\\nu$',fontsize=13)
    ax.set(xlim=(-.4,3.8),ylim=(-.22,1.38));ax.set_aspect('equal');ax.axis('off')
    fig.savefig(out,dpi=220);plt.close(fig)


class ResidualBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GELU(),
                                  nn.Conv2d(c,c,3,padding=1))
    def forward(self,x):return F.gelu(x+self.layers(x))


class FvmPIRN(nn.Module):
    def __init__(self,template,c=12):
        super().__init__();self.c=c
        self.fc=nn.Sequential(nn.Linear(1,64),nn.GELU(),nn.Linear(64,c*8*8))
        self.u=nn.Sequential(*[ResidualBlock(c) for _ in range(3)],nn.Conv2d(c,1,3,padding=1))
        self.v=nn.Sequential(*[ResidualBlock(c) for _ in range(3)],nn.Conv2d(c,1,3,padding=1))
        self.p=nn.Sequential(*[ResidualBlock(c) for _ in range(2)],nn.Conv2d(c,1,3,padding=1))
        self.register_buffer('template',torch.tensor(template,dtype=torch.float32))
    def forward(self,re):
        x=self.fc((re-15)/10).reshape(-1,self.c,8,8)
        x=F.interpolate(x,size=(64,64),mode='bilinear',align_corners=False)
        return self.template+torch.cat((self.u(x),self.v(x),self.p(x)),dim=1)


def load():
    z=np.load(ROOT/'fvm-reference'/'cavity_fvm_101_N64.npz')
    re=z['Re'];fields=z['fields'].copy()
    fields[:,2]-=fields[:,2].mean(axis=(1,2),keepdims=True)
    speed=(re/10)[:,None,None]
    fields[:,0]/=speed;fields[:,1]/=speed;fields[:,2]/=speed**2
    scale=np.sqrt(np.mean(fields**2,axis=(0,2,3),keepdims=True)).astype('float32')
    y=fields/scale
    return re,y.astype('float32'),scale


def train(epochs=400):
    torch.manual_seed(20261001);np.random.seed(20261001)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    re,y,scale=load()
    model=FvmPIRN(y.mean(axis=0,keepdims=True))
    x=torch.tensor(re[:,None]);target=torch.tensor(y)
    opt=torch.optim.Adam(model.parameters(),lr=.002)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=5e-5)
    history=[]
    for epoch in range(epochs):
        model.train();indices=torch.randperm(len(x));batch_losses=[]
        for ids in indices.split(16):
            pred=model(x[ids]);loss=F.mse_loss(pred,target[ids])
            opt.zero_grad();loss.backward();opt.step();batch_losses.append(loss.item())
        scheduler.step();history.append(float(np.mean(batch_losses)))
        if (epoch+1)%50==0:print(f'epoch {epoch+1}/{epochs}: MSE={history[-1]:.8f}',flush=True)
    return model,re,scale,history


def predict(model,re,scale):
    model.eval()
    with torch.no_grad():a=model(torch.tensor([[re]],dtype=torch.float32)).numpy()[0]*scale[0]
    speed=re/10
    a[0]*=speed;a[1]*=speed;a[2]*=speed**2;a[2]-=a[2].mean()
    return a


def reference(re):
    z=np.load(ROOT/'fvm-reference'/f'Re_{re:g}_N64.npz')
    a=np.stack([z[k] for k in ('u','v','p')]).astype('float32')
    a[2]-=a[2].mean()
    return a


def score(pred,ref):
    result={}
    for j,key in enumerate(('u','v','p')):
        d=pred[j]-ref[j];span=np.ptp(ref[j]);denom=float(np.max(ref[j]))
        if denom<=0:raise ValueError(f'Original-project error denominator nonpositive for {key}')
        original_error=np.abs(d)/denom
        result[key]={'relative_L2_full':float(np.linalg.norm(d)/np.linalg.norm(ref[j])),
                     'relative_L2_core_5':float(np.linalg.norm(d[5:-5,5:-5])/np.linalg.norm(ref[j,5:-5,5:-5])),
                     'NRMSE_over_range':float(np.sqrt(np.mean(d*d))/span),
                     'max_abs_error':float(np.max(np.abs(d))),
                     'original_error_denominator_ref_max':denom,
                     'original_error_mean':float(original_error.mean()),
                     'original_error_median':float(np.median(original_error)),
                     'original_error_max':float(original_error.max())}
    return result


def draw_case(re,pred,ref,path):
    label={15:'training',20.1:'interpolation',30:'extrapolation'}[re]
    fig,ax=plt.subplots(3,4,figsize=(14,9),constrained_layout=True)
    for j,name in enumerate(('u','v','p')):
        if name=='p':
            limit=float(np.percentile(np.abs(ref[j]),99))
            lo,hi=-limit,limit
        else:
            lo=min(pred[j].min(),ref[j].min());hi=max(pred[j].max(),ref[j].max())
        # Matches the original PIRN project's predict notebooks:
        # err = abs((reference - prediction) / reference.max()).
        denominator=float(np.max(ref[j]))
        if denominator<=0:raise ValueError(f'Nonpositive reference maximum for {name}')
        err=100*abs(pred[j]-ref[j])/denominator
        for c,a,title in ((0,pred[j],'PIRN prediction'),(1,ref[j],'FVM reference')):
            im=ax[j,c].imshow(a,origin='lower',extent=(0,.1,0,.1),cmap='coolwarm',vmin=lo,vmax=hi)
            fig.colorbar(im,ax=ax[j,c],fraction=.046,pad=.02)
            ax[j,c].set_title(title+(' (p color clipped)' if name=='p' else ''))
        im=ax[j,2].imshow(err,origin='lower',extent=(0,.1,0,.1),cmap='magma',vmin=0,
                          vmax=max(.1,float(np.max(err))))
        fig.colorbar(im,ax=ax[j,2],fraction=.046,pad=.02)
        ax[j,2].set_title('Absolute error / reference max (%)')
        ax[j,3].boxplot(err.ravel(),showfliers=False)
        ax[j,3].set(title=f'Error distribution (max {np.max(err):.2f}%)',ylabel='% of reference max',xticks=[])
        for c in range(3):ax[j,c].set(xlabel='x (m)',ylabel=f'{name}; y (m)',aspect='equal')
        for a in ax[j]:
            for side in ('top','right','bottom','left'):a.spines[side].set_visible(True)
    fig.suptitle(f'FVM-only cavity | Re={re:g} | {label} | pressure display clipped at reference 99th percentile',fontsize=14)
    fig.savefig(path,dpi=220);plt.close(fig)


def main():
    for r in (15,20.1,30):
        if not (ROOT/'fvm-reference'/f'Re_{r:g}_N64.npz').exists():
            raise FileNotFoundError(f'Missing independent FVM reference for Re={r}')
    OUT.mkdir(exist_ok=True)
    model,re,scale,history=train()
    torch.save({'state_dict':model.state_dict(),'scale':scale,'training_Re':re,
                'seed':20261001,'data_source':'FVM only, N=64'},OUT/'fvm_pirn.pth')
    report={'original_project_error_formula':'abs((reference - prediction) / reference.max())',
            'training':{'count':len(re),'Re_min':float(re.min()),'Re_max':float(re.max()),
                        'epochs':len(history),'final_MSE':history[-1],
                        'data':'cavity_fvm_101_N64.npz'},'cases':{}}
    for r in (15,20.1,30):
        pred=predict(model,r,scale);ref=reference(r)
        np.savez_compressed(OUT/f'Re_{r:g}_prediction_vs_FVM.npz',Re=r,prediction=pred,reference=ref)
        draw_case(r,pred,ref,OUT/f'Figure9_FVM_original_error_Re_{r:g}.png')
        report['cases'][str(r)]=score(pred,ref)
        print(f'Re={r:g}: {report["cases"][str(r)]}',flush=True)
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(6,3));ax.semilogy(history);ax.set(xlabel='Epoch',ylabel='MSE',title='FVM-only PIRN training')
    for side in ('top','right'):ax.spines[side].set_visible(True)
    fig.tight_layout();fig.savefig(OUT/'training_curve.png',dpi=200);plt.close(fig)
    draw_schematic(OUT/'Figure9a_schematic.png')
    ims=[Image.open(OUT/'Figure9a_schematic.png').convert('RGB')]+[Image.open(OUT/f'Figure9_FVM_original_error_Re_{r:g}.png').convert('RGB') for r in (15,20.1,30)]
    width=max(im.width for im in ims)
    combined=Image.new('RGB',(width,sum(im.height for im in ims)),'white');y=0
    for im in ims:combined.paste(im,((width-im.width)//2,y));y+=im.height
    combined.save(OUT/'Figure9_FVM_original_error_all_cases.png',optimize=True)


if __name__=='__main__':main()
