"""Steady 2-D lid-driven cavity: staggered finite-volume projection solver.

Cell-centred pressure, face-centred velocities, central flux/diffusion stencils,
explicit momentum step, and a discrete Neumann pressure-correction solve.  All
equations are solved dimensionlessly on the unit square, then u,v and kinematic
pressure are scaled to the manuscript's L=0.1 m and nu=0.01 m^2/s.

This is an independent solver; it does not reconstruct OpenFOAM's original mesh.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.fft import dctn, idctn
from scipy.interpolate import RegularGridInterpolator


def pressure_correction(rhs, h):
    n=rhs.shape[0]
    k=np.arange(n)
    lam=(2*np.cos(np.pi*k/n)-2)/h**2
    eig=lam[:,None]+lam[None,:]
    spectrum=dctn(rhs-rhs.mean(),type=2,norm='ortho')
    spectrum[0,0]=0
    eig[0,0]=1
    phi=idctn(spectrum/eig,type=2,norm='ortho')
    return phi-phi.mean()


def solve(re=15., n=64, tol=1e-7, max_steps=30000):
    h=1/n
    dt=min(.16*re*h*h,.30*h)
    u=np.zeros((n,n+1)); v=np.zeros((n+1,n));p=np.zeros((n,n))
    last_u=u.copy();last_v=v.copy();last_p=p.copy()
    for step in range(1,max_steps+1):
        # Tangential wall velocities enter through ghost face values.
        ue=np.empty((n+2,n+1));ue[1:-1]=u;ue[0]=-u[0];ue[-1]=2-u[-1]
        ve=np.empty((n+1,n+2));ve[:,1:-1]=v;ve[:,0]=-v[:,0];ve[:,-1]=-v[:,-1]
        ui=u[:,1:-1];vi=v[1:-1,:]
        lap_u=(u[:,2:]+u[:,:-2]+ue[2:,1:-1]+ue[:-2,1:-1]-4*ui)/h**2
        lap_v=(v[2:,:]+v[:-2,:]+ve[1:-1,2:]+ve[1:-1,:-2]-4*vi)/h**2
        # Conservative flux divergence through each staggered control volume.
        u_e=(ui+u[:,2:])/2;u_w=(u[:,:-2]+ui)/2
        v_n_at_u=(v[1:,:-1]+v[1:,1:])/2
        v_s_at_u=(v[:-1,:-1]+v[:-1,1:])/2
        u_n=(ui+ue[2:,1:-1])/2;u_s=(ui+ue[:-2,1:-1])/2
        conv_u=(u_e**2-u_w**2+v_n_at_u*u_n-v_s_at_u*u_s)/h
        u_e_at_v=(u[:-1,1:]+u[1:,1:])/2
        u_w_at_v=(u[:-1,:-1]+u[1:,:-1])/2
        v_e=(vi+ve[1:-1,2:])/2;v_w=(vi+ve[1:-1,:-2])/2
        v_n=(vi+v[2:,:])/2;v_s=(v[:-2,:]+vi)/2
        conv_v=(u_e_at_v*v_e-u_w_at_v*v_w+v_n**2-v_s**2)/h
        us=u.copy();vs=v.copy()
        us[:,1:-1]+=dt*(-conv_u+lap_u/re-(p[:,1:]-p[:,:-1])/h)
        vs[1:-1,:]+=dt*(-conv_v+lap_v/re-(p[1:,:]-p[:-1,:])/h)
        divergence=(us[:,1:]-us[:,:-1]+vs[1:,:]-vs[:-1,:])/h
        phi=pressure_correction(divergence/dt,h)
        u[:,1:-1]=us[:,1:-1]-dt*(phi[:,1:]-phi[:,:-1])/h
        v[1:-1,:]=vs[1:-1,:]-dt*(phi[1:,:]-phi[:-1,:])/h
        p+=phi;p-=p.mean()
        if step%200==0:
            change=max(np.max(abs(u-last_u)),np.max(abs(v-last_v)))
            pchange=np.max(abs(p-last_p))
            if change<tol and pchange<10*tol:
                break
            last_u[:]=u;last_v[:]=v;last_p[:]=p
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise RuntimeError(f'Non-finite flow at step {step}; reduce timestep')
    else:
        raise RuntimeError(f'Not converged after {max_steps} steps; last change={change:g}, pressure change={pchange:g}')
    div=(u[:,1:]-u[:,:-1]+v[1:,:]-v[:-1,:])/h
    return u,v,p,{'steps':step,'dt':dt,'last_200_step_velocity_change':float(change),
                   'last_200_step_pressure_change':float(pchange),
                   'max_dimensionless_divergence':float(np.max(np.abs(div)))}


def to_vertex_grid(u,v,p,size=64):
    """Resample staggered variables to the 64x64 plotting grid of pre-data."""
    n=p.shape[0]; q=np.linspace(0,1,size); yy,xx=np.meshgrid(q,q,indexing='ij')
    points=np.stack((yy,xx),axis=-1)
    centers=(np.arange(n)+.5)/n; faces=np.arange(n+1)/n
    ue=np.zeros((n+2,n+1));ue[1:-1]=u;ue[-1,1:-1]=1
    ve=np.zeros((n+1,n+2));ve[:,1:-1]=v
    ug=RegularGridInterpolator((np.r_[0,centers,1],faces),ue)(points)
    vg=RegularGridInterpolator((faces,np.r_[0,centers,1]),ve)(points)
    pg=RegularGridInterpolator((centers,centers),p,bounds_error=False,fill_value=None)(points)
    return ug,vg,pg-pg.mean()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--re',type=float,nargs='+',default=[15.])
    ap.add_argument('--n',type=int,default=64)
    ap.add_argument('--out',type=Path,default=Path('fvm-reference'))
    args=ap.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    for re in args.re:
        u,v,p,diag=solve(re,args.n)
        speed=re/10 # Re = U*0.1/0.01
        ug,vg,pg=to_vertex_grid(u,v,p)
        path=args.out/f'Re_{re:g}_N{args.n}.npz'
        np.savez_compressed(path,Re=re,N=args.n,u=ug*speed,v=vg*speed,p=pg*speed**2,
                            u_faces=u*speed,v_faces=v*speed,p_cells=p*speed**2,
                            diagnostics=np.array(str(diag)))
        print(f'Re={re:g}, N={args.n}: {diag}; saved {path}',flush=True)


if __name__=='__main__':main()
