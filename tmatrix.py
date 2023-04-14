#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 14:47:17 2021

@author: semmerson
"""

# Script for generating T-matrix lookup tables for passive radar simulations

from pytmatrix.tmatrix import Scatterer
from pytmatrix.psd import PSDIntegrator, GammaPSD, ExponentialPSD, UnnormalizedGammaPSD
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

def snow_ar(D_eq):
    return 0.856*D_eq**-0.113

p_dict = {}
p_dict['dry_hail'] = {}
p_dict['dry_hail'][103.1] = 3.17-0.001j
p_dict['dry_hail'][54.1] = 3.17-0.001j
p_dict['dry_hail'][31.4] = 3.17-0.002j
p_dict['hail_a'] = {}
p_dict['hail_a'][103.1] = 3.0316 - 0.00102j
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
p_dict['dry_snow'] = {}
p_dict['dry_snow'][103.1] = 1.39-1e-6j
p_dict['dry_snow'][54.1] = 1.39-1e-6j
p_dict['dry_snow'][31.4] = 1.39-1e-6j

ptype = 'hail'
Lambda = 54.1
ep = p_dict[ptype][Lambda]
m = np.sqrt((np.abs(ep)+np.real(ep))/2) + np.sqrt((np.abs(ep)-np.real(ep))/2)*1j
nsAz = 361
saz = np.linspace(0,180,nsAz)
nr = 240
if ptype == 'rain':
    rads = np.linspace(0.1,4.0,nr)
else:
    rads = np.linspace(1,(nr+3)/4,nr)
ize = 90.0
sze = 90.0
iaz = 0.0
# #saz,rads = np.meshgrid(saz,rads)

# # scatterer.psd_integrator = PSDIntegrator()
# # if ptype == 'rain':
# #     scatterer.psd_integrator.axis_ratio_func = lambda D: 1.0/drop_ar(D)
# # elif ptype == 'dry_snow':
# #     scatterer.psd_integrator.axis_ratio_func = lambda D: 1.0/snow_ar(D)
# # else:
# #     scatterer.psd_integrator.axis_ratio_func = lambda D: 1.0/hail_ar(D)


# #scatterer.psd_integrator.axis_ratio_func = lambda D: 0.8
# # Dm = 40.0
# # scatterer.psd_integrator.D_max = Dm
# # scatterer.psd_integrator.num_points = 32
# # scatterer.psd_integrator.geometries = ()

# sh = np.zeros((nsAz,nr))
# sv = np.zeros((nsAz,nr))
# dco = np.zeros((nsAz,nr))
# dh = np.zeros((nsAz,nr))
# dv = np.zeros((nsAz,nr))
# for j,rad in enumerate(rads):
#     scatterer = Scatterer(radius=rad,wavelength=Lambda, m=m, ndgs=3, axis_ratio=1/hail_ar(2*rad))
#     scatterer.set_geometry(tmatrix_aux.geom_horiz_forw) 
#     for i,az in enumerate(saz):
#         geom = (ize, sze, 0.0, az, 0.0, 0.0)
#         # geom = tmatrix_aux.geom_horiz_back
#         scatterer.set_geometry(geom)
#         sh[i,j] = radar.refl(scatterer)
#         sv[i,j] = radar.refl(scatterer,False)
#         dco[i,j] = radar.delta_hv(scatterer)
#         S = scatterer.get_S()
#         dv[i,j] = np.angle(S[0,0])
#         dh[i,j] = np.angle(S[1,1])
# #     # refh[it.multi_index] = 10*np.log10(radar.refl(scatterer))
# #     # zdr[idx] = radar.Zdr(scatterer)
# #     # rho[it.multi_index] = radar.rho_hv(scatterer)
# sh = np.concatenate((sh,sh[:0:-1,:]),axis=0)
# sv = np.concatenate((sv,sv[:0:-1,:]),axis=0)
# dco = np.concatenate((dco,dco[:0:-1,:]),axis=0)
# dv = np.concatenate((dv,dv[:0:-1,:]),axis=0)
# dh = np.concatenate((dh,dh[:0:-1,:]),axis=0)


nsAz = 46
nsZe = 46
niZe = 46
nl = 46
nn0 = 36
# if ptype == 'rain':
#     rads = np.linspace(0.1,4.0,nr)
# else:
#     rads = np.linspace(1,nr,nr)
# svv = np.zeros((nn0,nl,niZe,nsZe,nsAz)).astype('complex64')
# shh = np.zeros((nn0,nl,niZe,nsZe,nsAz)).astype('complex64')
# svh = np.zeros((nn0,nl,niZe,nsZe,nsAz)).astype('complex64')
# shv = np.zeros((nn0,nl,niZe,nsZe,nsAz)).astype('complex64')
sv = np.zeros((nn0,nl,niZe,nsZe,nsAz)).astype('complex64')
sh = np.zeros((nn0,nl,niZe,nsZe,nsAz)).astype('complex64')
Av = np.zeros((nn0,nl))
Ah = np.zeros((nn0,nl))
Kv = np.zeros((nn0,nl))
Kh = np.zeros((nn0,nl))
cdr = np.zeros((nn0,nl))
lambs = 10**np.linspace(-0.5,1.7,nl)
n0s = 10**np.linspace(-10,7,nn0)
lambs,n0s = np.meshgrid(lambs,n0s)
iter0 = np.nditer(lambs,flags=['multi_index'])
scatterer = Scatterer(wavelength=Lambda, m=m, ndgs=3)
scatterer.psd_integrator = PSDIntegrator()
if ptype == 'rain':
    scatterer.psd_integrator.axis_ratio_func = lambda D: 1.0/drop_ar(D)
elif ptype == 'dry_snow':
    scatterer.psd_integrator.axis_ratio_func = lambda D: 1.0/snow_ar(D)
else:
    scatterer.psd_integrator.axis_ratio_func = lambda D: 1.0/hail_ar(D)


#scatterer.psd_integrator.axis_ratio_func = lambda D: 0.8
if ptype in ['rain','snow']:
    Dm = 8.0
else:
    Dm = 40.0
scatterer.psd_integrator.D_max = Dm
scatterer.psd_integrator.num_points = 32
scatterer.psd_integrator.geometries = ()
ize = np.linspace(0,90,niZe)+90
sze = np.linspace(0,180,nsZe)
saz = np.linspace(0,180,nsAz)
sze,ize,saz = np.meshgrid(sze,ize,saz)
iter1 = np.nditer(sze,flags=['multi_index'])
for x in iter1:
    scatterer.psd_integrator.geometries += ((ize[iter1.multi_index], sze[iter1.multi_index], 0.0, saz[iter1.multi_index], 0.0, 0.0),)
scatterer.psd_integrator.geometries += (tmatrix_aux.geom_horiz_forw,)
scatterer.psd_integrator.init_scatter_table(scatterer,verbose=False)
scatterer.set_geometry(tmatrix_aux.geom_horiz_forw)
for y in iter0:
    print(np.log10(lambs[iter0.multi_index]),np.log10(n0s[iter0.multi_index]))
    scatterer.psd = ExponentialPSD(N0 = n0s[iter0.multi_index],Lambda=lambs[iter0.multi_index],D_max=Dm)
    #scatterer.psd = UnnormalizedGammaPSD(N0 = 10**5,Lambda=10,D_max=Dm)
    #geometries = ()
    
    #             geometries += ((k*22.5+90, j, 0.0, i, 0.0, 0.0),)
    scatterer.or_pdf = orientation.gaussian_pdf(std=10.0)
    
    S = scatterer.get_S()
#     Z = scatterer.get_Z()
#     cdr[iter0.multi_index] = (Z[0,0]+Z[3,0]+Z[0,3]+Z[3,3])/(2*(Z[0,0]+Z[3,0]))
    Kh[iter0.multi_index] = 1e-3*180/np.pi*scatterer.wavelength/1e3*S[1,1].real
    Kv[iter0.multi_index] = 1e-3*180/np.pi*scatterer.wavelength/1e3*S[0,0].real
    Ah[iter0.multi_index] = 8.686e-3*scatterer.wavelength/1e3*S[1,1].imag
    Av[iter0.multi_index] = 8.686e-3*scatterer.wavelength/1e3*S[0,0].imag
    
# # scatterer.orient = orientation.orient_averaged_fixed
# # start = timer()

# # end = timer()
# # time = end-start
# # print(time)
# #

    # iter2 = np.nditer(sze,flags=['multi_index'])
    # for x in iter2:
    #     geom = (ize[iter2.multi_index], sze[iter2.multi_index], 0.0, saz[iter2.multi_index], 0.0, 0.0)
    #     # geom = tmatrix_aux.geom_horiz_back
    #     scatterer.set_geometry(geom)
    #     idx = iter0.multi_index + iter2.multi_index
    
    # # refh[it.multi_index] = 10*np.log10(radar.refl(scatterer))
    # # zdr[idx] = radar.Zdr(scatterer)
    # # rho[it.multi_index] = radar.rho_hv(scatterer)
    #     sh[idx] = radar.refl(scatterer)
    #     sv[idx] = radar.refl(scatterer,False)
        # S = scatterer.get_S()
        # shh[idx] = S[1,1]
        # shv[idx] = S[1,0]
        # svh[idx] = S[0,1]
        # svv[idx] = S[0,0]
        #Z = scatterer.get_Z()
        # sh[idx] = np.arctan2(Z[1,3]-Z[0,3],Z[0,2]-Z[1,2])
        #sh[idx] = (Z[0,0]+Z[3,0]+Z[0,3]+Z[3,3])/(2*(Z[0,0]+Z[3,0]))
        # sv[idx] = np.arctan2(Z[1,3]+Z[0,3],Z[0,2]+Z[1,2])
        

# nsh = np.concatenate((sh,sh[:,:,:,:,:0:-1]),axis=4).T
# nsv = np.concatenate((sv,sv[:,:,:,:,:0:-1]),axis=4).T
# nshh = np.concatenate((shh,shh[:,:,:,:,:0:-1]),axis=4).T
# nsvv = np.concatenate((svv,svv[:,:,:,:,:0:-1]),axis=4).T
# nshv = np.concatenate((shv,shv[:,:,:,:,:0:-1]),axis=4).T
# nsvh = np.concatenate((svh,svh[:,:,:,:,:0:-1]),axis=4).T
# nsh = nshh.astype('float32')
# nsv = nsvv.astype('float32')

# # phi = np.linspace(0,180,nsZe)*np.pi/180
# # theta = np.linspace(0,360,nsAz*2-1)*np.pi/180
# # phi,theta = np.meshgrid(phi,theta)




# # a = 0
# # d = 39
# # nsv = nsv[:,:,a,d]
# # nsh = nsh[:,:,a,d]
# # nsv = np.log10(nsv)
# # nsh = np.log10(nsh)
# # nsv[np.abs(nsv) > 15] = np.nan
# # nsh[np.abs(nsh) > 15] = np.nan

# # x = nsh*np.cos(theta)*np.sin(phi)
# # y = nsh*np.sin(theta)*np.sin(phi)
# # z = nsh*np.cos(phi)

# ref = (np.linspace(0,2*np.pi,91),np.linspace(0,np.pi,46),np.linspace(0,np.pi/2,46),np.log10(n0s[:,0]),np.log10(lambs[0,:]))
# with lzma.open(f'{scatterer.wavelength}mm_S_{ptype}_New.xz','wb') as f:
#     pickle.dump(ref,f)
#     pickle.dump(nsv,f)
#     pickle.dump(nsh,f)
    # pickle.dump(nsvv,f)
    # pickle.dump(nsvh,f)
    # pickle.dump(nshv,f)
    # pickle.dump(nshh,f)

ref = (np.log10(n0s[:,0]),np.log10(lambs[0,:]))
with lzma.open(f'{scatterer.wavelength}mm_A_{ptype}.xz','wb') as f:
    pickle.dump(ref,f)
    pickle.dump(Av,f)
    pickle.dump(Ah,f)

ref = (np.log10(n0s[:,0]),np.log10(lambs[0,:]))
with lzma.open(f'{scatterer.wavelength}mm_K_{ptype}.xz','wb') as f:
    pickle.dump(ref,f)
    pickle.dump(Kv,f)
    pickle.dump(Kh,f)
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



