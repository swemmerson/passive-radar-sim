#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 14:47:17 2021

@author: semmerson
"""

# Script for generating T-matrix lookup tables for passive radar simulations

from pytmatrix.tmatrix import Scatterer
from pytmatrix.psd import PSDIntegrator, GammaPSD, ExponentialPSD
from pytmatrix import orientation, radar, tmatrix_aux, refractive
import numpy as np
import lzma,pickle

def drop_ar(D_eq):
    if D_eq < 0.7:
        return 1.0;
    elif D_eq < 1.5:
        return 1.173 - 0.5165*D_eq + 0.4698*D_eq**2 - 0.1317*D_eq**3 - \
            8.5e-3*D_eq**4
    else:
        return 1.065 - 6.25e-2*D_eq - 3.99e-3*D_eq**2 + 7.66e-4*D_eq**3 - \
            4.095e-5*D_eq**4 


def hail_ar(D_eq):
    return 1.0/(1.1749+0.9697/D_eq)

p_dict = {}
p_dict['dry_hail'] = {}
p_dict['dry_hail'][103.1] = 3.17-0.001j
p_dict['dry_hail'][54.1] = 3.17-0.001j
p_dict['dry_hail'][31.4] = 3.17-0.002j
p_dict['hail'] = {}
p_dict['hail'][103.1] = 8.63-1.63j
p_dict['hail'][54.1] = 7.52-2.58j
p_dict['hail'][31.4] = 6.35-2.73j
p_dict['wet_hail'] = {}
p_dict['wet_hail'][103.1] = 20.71-5.23j
p_dict['wet_hail'][54.1] = 17.14-8.30j
p_dict['wet_hail'][31.4] = 13.38-8.80j
p_dict['rain'] = {}
p_dict['rain'][103.1] = 80.89-19.92j
p_dict['rain'][54.1] = 68.38-33.45j
p_dict['rain'][31.4] = 47.92-39.55j

ptype = 'hail'
Lambda = 31.4
ep = p_dict[ptype][Lambda]
m = np.sqrt((np.abs(ep)+np.real(ep))/2) + np.sqrt((np.abs(ep)-np.real(ep))/2)*1j

nsAz = 91
nsZe = 91
niZe = 46
nr = 40
if ptype == 'rain':
    rads = np.linspace(0.1,4.0,nr)
else:
    rads = np.linspace(1,nr,nr)
sv = np.zeros((nr,niZe,nsZe,nsAz))
sh = np.zeros((nr,niZe,nsZe,nsAz))
# zdr = np.zeros((nr,niZe,nsZe,nsAz))
for n,r in enumerate(rads):
    print(r)
    if ptype == 'rain':
        ar = 1.0/drop_ar(2*r)
    else:
        ar = 1.0/hail_ar(2*r)
    scatterer = Scatterer(radius=r,axis_ratio=0.6,wavelength=Lambda, m=m, ndgs=3)
    #scatterer.psd_integrator = PSDIntegrator()
    #scatterer.psd_integrator.axis_ratio_func = lambda D: 1.0/drop_ar(D)
    #scatterer.psd_integrator.axis_ratio_func = lambda D: 0.8
    #scatterer.psd_integrator.D_max = 100.0
    #scatterer.psd_integrator.num_points = 16
    #scatterer.psd_integrator.geometries = ()
    geometries = ()
    
    ize = np.linspace(0,90,niZe)+90
    sze = np.linspace(0,180,nsZe)
    saz = np.linspace(0,180,nsAz)
    sze,ize,saz = np.meshgrid(sze,ize,saz)
    
    # for i in range(nsAz):
    #     for j in range(nsZe):
    #         for k in range(niZe):
    #             print(i,j,k)
    #             #scatterer.psd_integrator.geometries += ((k*22.5+90, j, 0.0, i, 0.0, 0.0),)
    #             geometries += ((k*22.5+90, j, 0.0, i, 0.0, 0.0),)
    scatterer.or_pdf = orientation.gaussian_pdf(std=30.0)
    # scatterer.orient = orientation.orient_averaged_fixed
    # start = timer()
    # scatterer.psd_integrator.init_scatter_table(scatterer,verbose=True)
    # end = timer()
    # time = end-start
    # print(time)
    #scatterer.psd = ExponentialPSD(N0 = 1e2,Lambda= 0.025,D_max=100.0)
    
    it = np.nditer(sze,flags=['multi_index'])
    for x in it:
        geom = (ize[it.multi_index], sze[it.multi_index], 0.0, saz[it.multi_index], 0.0, 0.0)
        scatterer.set_geometry(geom)
        idx = (n,) + it.multi_index
        # refh[it.multi_index] = 10*np.log10(radar.refl(scatterer))
        # zdr[idx] = radar.Zdr(scatterer)
        # rho[it.multi_index] = radar.rho_hv(scatterer)
        sh[idx] = 4*np.pi*np.abs(scatterer.get_S()[1,1])**2
        sv[idx] = 4*np.pi*np.abs(scatterer.get_S()[0,0])**2

nsh = np.concatenate((sh,sh[:,:,:,:0:-1]),axis=3).T
nsv= np.concatenate((sv,sv[:,:,:,:0:-1]),axis=3).T



phi = np.linspace(0,180,nsZe)*np.pi/180
theta = np.linspace(0,360,nsAz*2-1)*np.pi/180
phi,theta = np.meshgrid(phi,theta)




a = 0
d = 39
nsv = nsv[:,:,a,d]
nsh = nsh[:,:,a,d]
nsv = np.log10(nsv)
nsh = np.log10(nsh)
nsv[np.abs(nsv) > 15] = np.nan
nsh[np.abs(nsh) > 15] = np.nan

# x = nsh*np.cos(theta)*np.sin(phi)
# y = nsh*np.sin(theta)*np.sin(phi)
# z = nsh*np.cos(phi)

ref = (np.linspace(0,2*np.pi,181),np.linspace(0,np.pi,91),np.linspace(0,np.pi/2,46),rads)
with lzma.open(f'{scatterer.wavelength}mm_S_{ptype}.xz','wb') as f:
    pickle.dump(ref,f)
    pickle.dump(nsv,f)
    pickle.dump(nsh,f)
# X = x
# Y = y
# Z = z
# r = np.nanmax(nsh)*1.5
# x = nrho*np.cos(theta)*np.sin(phi)
# y = nrho*np.sin(theta)*np.sin(phi)
# z = nrho*np.cos(phi)
# fig = make_subplots(rows=1, cols=1,
#     specs=[[{'type': 'surface'}]])
# fig.add_trace(go.Surface(x=X, y=Y, z=Z, surfacecolor=np.sqrt(X**2 + Y**2 + Z**2),colorscale='jet'),row=1,col=1)

# # Z = np.sqrt(x**2 + y**2 + z**2)
# # col = pyart.graph.cm.NWSRef(Z/np.max(Z))
# # fig = plt.figure(figsize=(8,8))
# # ax = fig.add_subplot(111, projection='3d')
# # ax.plot_surface(x,y,z,facecolors=col)
# # ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

# # a = 0
# # def animate(t):
# #     plt.clf()  
# #     X = x[:,:,a,t]
# #     Y = y[:,:,a,t]
# #     Z = z[:,:,a,t]
# #     zz = np.sqrt(X**2 + Y**2 + Z**2)
# #     col = pyart.graph.cm.NWSRef(zz/np.max(zz))
# #     ax = fig.add_subplot(111, projection='3d')
# #     ax.plot_surface(X,Y,Z,facecolors=col)
# #     ax.set_xlim((np.min(x),np.max(x)))
# #     ax.set_ylim((np.min(y),np.max(y)))
# #     ax.set_zlim((np.min(z),np.max(z)))
# #     ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
# # fig = plt.figure(figsize=(8,8))
# # anim = mani.FuncAnimation(fig, animate, interval=1, frames=nr)
# # anim.save('test1.gif',writer='imagemagick',fps=5)
# # fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=np.sqrt(x**2 + y**2 + z**2),colorscale='jet'),row=1,col=1)

# fig.add_trace(go.Scatter3d(
#     x=[0, r*np.cos(0)*np.sin(ize[a,0,0]*np.pi/180)],
#     y=[0, r*np.sin(0)*np.sin(ize[a,0,0]*np.pi/180)],
#     z=[0, r*np.cos(ize[a,0,0]*np.pi/180)],
#     name="z",
# ))
# fig.update_layout(
#     title_text=f'Shh @{np.round(scatterer.wavelength/10,2)} cm',
#     scene=dict(
#                   aspectmode='data'
#           ),
# )
# plot(fig)


# def tmatrixWeights(table,ref,r,ize,sze,saz):
#     r = r*np.ones(ize.shape)
#     weights = interpolate.interpn(ref,table,np.vstack([saz,sze,ize,r]).T)
#     return weights


# r = 2.33
# n = int(1e7)
# izen = np.random.uniform(0,50,n)*np.pi/180
# szen = np.random.uniform(90,120,n)*np.pi/180
# sazn = np.random.uniform(0,360,n)*np.pi/180
# start = timer()
# wv = tmatrixWeights(nsv,ref,r,izen,szen,sazn)
# end = timer()
# time = end-start
# print(time)



