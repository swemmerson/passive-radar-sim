#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:53:46 2020

@author: semmerson
"""
import os, pyart, pickle, lzma
import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
import matplotlib.animation as mani
import datetime as dt
from netCDF4 import Dataset
from scipy.fftpack import fft2, fftshift
from scipy import interpolate, ndimage, signal
from skimage.restoration import unwrap_phase
from scipy.io import loadmat
from numba import njit
from line_profiler import LineProfiler
#from numba.types import float64, int64, complex128

def makeRadarStruct():
    radar = {}
    radar['txPos'] = np.array([0,-15,0])*1e3 #Tx position
    # x = np.linspace(-30,0,4)
    # y = np.linspace(-30,0,4)
    # x,y = np.meshgrid(x,y)
    # z = np.zeros((4,4))
    #radar['rxPos'] = np.reshape(np.stack((x,y,z)),(3,16)).T*1e3+radar['txPos']
    radar['rxPos'] = np.array([[-13,-7.5,0],[13,-7.5,0],[8.66,-10,0],[-8.66,-10,0],[4.33,-12.5,0],[-4.33,-12.5,0],[0,-15,0]])*1e3 #Rx position 
    radar['lambda'] = 0.05415 #Operating wavelength
    radar['prt'] = 1/4000  #Tx PRT (s)
    radar['tau'] = 1e-6 #Pulse width (s)
    radar['fs'] = 1/radar['tau'] #Sampling Rate samples/s
    radar['Pt'] = 250e3 #Peak transmit power (W)
    radar['Tant'] = 63.36 #Antenna noise temp/brightness
    radar['Frx'] = 3 #Rx noise figure
    radar['M'] = 40 #Pulses per dwell
    radar['rxMechTilt'] = 1 #Mechanical tilt of Rx antenna
    radar['txMechEl'] = 0.5*np.arange(1,2) #Mechanical Tx elevation tilt
    radar['txMechAz'] = 1 #Baseline mechanical Tx position
    radar['txAz'] = np.mod(np.linspace(270,449,180),360) #Transmit azimuths
    radar['txEl'] = np.zeros(radar['txAz'].shape) #Electrical tx steering elv.
    radar['txG'] = 45.5 #Transmit antenna gain (dBi)
    radar['rxG'] = 48 #Receive antenna gain (dBi)
    radar['receiverGain'] = 45 #Receiver gain (dB)
    return radar

def makeWxStruct():
    wx = {}
    wx['scatMode'] = 'rayleigh' #Rayleigh or Bragg scattering
    wx['xSize'] = 30e3 #x,y,z dimensions of simulation volume (m)
    wx['ySize'] = 30e3
    wx['zSize'] = 4e3
    wx['getZh'] = 'Zh=35*np.ones((1,nPts))' #Zh/Zv values for uniform test case
    wx['getZv'] = 'Zv=Zh'              #not currently used
    wx['wrfDate'] = '2013-06-01 01:20:10' #NWP data query date: yyyy-mm-dd HH:MM:SS
    wx['wrfOffset'] = np.array([-14,4,0.3])#np.array([-20,-20,0.2]) # offset of simulation volume
                                               # from origin within NWP data                                           
    wx['spw'] = 2 #Spectrum width
    wx['ptsPerMinVol'] = 4 #Number of points in smallest resolution volume

    return wx

def datenum64(d):
    return 366 + d.item().toordinal() + (d.item() - dt.datetime.fromordinal(d.item().toordinal())).total_seconds()/(24*60*60)

# def getWrf(dstr,xq,yq,zq):
#     #Unit tested
#     f = loadmat('wrfDirectory.mat') #load NWP directory
#     YY = f['XX'][:]
#     XX = f['YY'][:]
#     ZZ = f['ZZ'][:]
#     dates = np.stack(f['dates'][:])[:,0]
#     uFiles = np.stack(f['uFiles'][:][0])[:,0]
#     vFiles = np.stack(f['vFiles'][:][0])[:,0]
#     wFiles = np.stack(f['wFiles'][:][0])[:,0]
#     zFiles = np.stack(f['zFiles'][:][0])[:,0]
#     dnumq = datenum64(np.datetime64(dstr)) #convert to datenum
#     #Get boundaries of NWP and query points
#     xmin,xmax = (np.min(XX),np.max(XX))
#     ymin,ymax = (np.min(YY),np.max(YY))
#     zmin,zmax = (np.min(ZZ),np.max(ZZ))
#     xqmin,xqmax = (np.min(xq),np.max(xq))
#     yqmin,yqmax = (np.min(yq),np.max(yq))
#     zqmin,zqmax = (np.min(zq),np.max(zq))
    
#     #Check if all query points fall within NWP volume
#     xgood = (xmin<xqmin) & (xmax>xqmax)
#     ygood = (ymin<yqmin) & (ymax>yqmax)
#     zgood = (zmin<zqmin) & (zmax>zqmax)
#     if xgood & ygood & zgood:
#         if np.any(dates==dnumq): #Interpolate directly if querying sampled instant
#             ind = np.where(dates==dnumq)
            
#             #Load variables from corresponding files and spatially
#             #interpolate
#             uFile = uFiles[ind]
#             vFile = vFiles[ind]
#             wFile = wFiles[ind]
#             zFile = zFiles[ind]
#             f = loadmat('WRF/'+uFile)
#             data_int = f['data_int'][:]
#             uu = interpolate.interpn((XX[:,0,0],YY[0,:,0],ZZ[0,0,:]), data_int,np.vstack((xq.flatten(),yq.flatten(),zq.flatten())))
#             f = loadmat('WRF/'+vFile)
#             data_int = f['data_int'][:]
#             vv = interpolate.interpn((XX,YY,ZZ), data_int,np.vstack((xq.flatten(),yq.flatten(),zq.flatten())))
#             f = loadmat('WRF/'+wFile)
#             data_int = f['data_int'][:]
#             ww = interpolate.interpn((XX,YY,ZZ), data_int,np.vstack((xq.flatten(),yq.flatten(),zq.flatten())))
#             f = loadmat('WRF/'+zFile)
#             data_int = f['data_int'][:]
#             zz = 10*np.log10(interpolate.interpn((XX,YY,ZZ), 10**(data_int/10),np.vstack((xq.flatten(),yq.flatten(),zq.flatten()))))
#         elif (dnumq > np.min(dates)) & (dnumq < np.max(dates)): #Otherwise interpolate in time
#             #Get variables from corresponding files at neighboring times
#             #samples
#             d0 = np.max(dates[dates<dnumq])
#             d1 = np.min(dates[dates>dnumq])
#             ind0 = np.where(dates==d0)[0]
#             ind1 = np.where(dates==d1)[0]
#             uFile0 = uFiles[ind0][0]
#             vFile0 = vFiles[ind0][0]
#             wFile0 = wFiles[ind0][0]
#             zFile0 = zFiles[ind0][0]
#             uFile1 = uFiles[ind1][0]
#             vFile1 = vFiles[ind1][0]
#             wFile1 = wFiles[ind1][0]
#             zFile1 = zFiles[ind1][0]
#             #Load variables from corresponding files and spatially
#             #interpolate
#             f = loadmat('WRF/'+uFile0)
#             data_int = f['data_int'][:]
#             rgi = interpolate.RegularGridInterpolator((XX[:,0,0],YY[0,:,0],ZZ[0,0,:]), np.swapaxes(data_int,0,1))
#             uu0 = rgi(np.array([xq,yq,zq]).T)
#             f = loadmat('WRF/'+vFile0)
#             data_int = f['data_int'][:]
#             rgi = interpolate.RegularGridInterpolator((XX[:,0,0],YY[0,:,0],ZZ[0,0,:]), np.swapaxes(data_int,0,1))
#             vv0 = rgi(np.array([xq,yq,zq]).T)
#             f = loadmat('WRF/'+wFile0)
#             data_int = f['data_int'][:]
#             rgi = interpolate.RegularGridInterpolator((XX[:,0,0],YY[0,:,0],ZZ[0,0,:]), np.swapaxes(data_int,0,1))
#             ww0 = rgi(np.array([xq,yq,zq]).T)
#             f = loadmat('WRF/'+zFile0)
#             data_int = f['data_int'][:]
#             rgi = interpolate.RegularGridInterpolator((XX[:,0,0],YY[0,:,0],ZZ[0,0,:]), np.swapaxes(10**(data_int/10),0,1))
#             zz0 = rgi(np.array([xq,yq,zq]).T)
            
#             f = loadmat('WRF/'+uFile1)
#             data_int = f['data_int'][:]
#             rgi = interpolate.RegularGridInterpolator((XX[:,0,0],YY[0,:,0],ZZ[0,0,:]), np.swapaxes(data_int,0,1))
#             uu1 = rgi(np.array([xq.flatten(),yq.flatten(),zq.flatten()]).T)
#             f = loadmat('WRF/'+vFile1)
#             data_int = f['data_int'][:]
#             rgi = interpolate.RegularGridInterpolator((XX[:,0,0],YY[0,:,0],ZZ[0,0,:]), np.swapaxes(data_int,0,1))
#             vv1 = rgi(np.array([xq.flatten(),yq.flatten(),zq.flatten()]).T)
#             f = loadmat('WRF/'+wFile1)
#             data_int = f['data_int'][:]
#             rgi = interpolate.RegularGridInterpolator((XX[:,0,0],YY[0,:,0],ZZ[0,0,:]), np.swapaxes(data_int,0,1))
#             ww1 = rgi(np.array([xq.flatten(),yq.flatten(),zq.flatten()]).T)
#             f = loadmat('WRF/'+zFile1)
#             data_int = f['data_int'][:]
#             rgi = interpolate.RegularGridInterpolator((XX[:,0,0],YY[0,:,0],ZZ[0,0,:]), np.swapaxes(10**(data_int/10),0,1))
#             zz1 = rgi(np.array([xq.flatten(),yq.flatten(),zq.flatten()]).T)
            
#             #Temporal interpolation
#             uu = uu0 + (dnumq-d0)*(uu1-uu0)/(d1-d0)
#             vv = vv0 + (dnumq-d0)*(vv1-vv0)/(d1-d0)
#             ww = ww0 + (dnumq-d0)*(ww1-ww0)/(d1-d0)
#             zz = 10*np.log10(zz0 + (dnumq-d0)*(zz1-zz0)/(d1-d0))
#             return (zz,uu,vv,ww)
#         else:
#             print('Specified date out of bounds.')
#     else:
#         print('Specified volume out of bounds.')

def getWrf(xq,yq,zq):
    #Unit tested
    fh = Dataset('may20.nc', 'r')
    XX = fh.variables['x'][:]
    YY = fh.variables['y'][:]
    ZZ = fh.variables['z'][:]
    #XX,YY,ZZ = np.meshgrid(XX,YY,ZZ)
    # UU = np.swapaxes(fh.variables['u'][:],0,1)
    # VV = np.swapaxes(fh.variables['v'][:],0,1)
    # WW = np.swapaxes(fh.variables['w'][:],0,1)
    # ref = np.swapaxes(fh.variables['reflectivity'][:],0,1)
    UU = fh.variables['u'][:]
    VV = fh.variables['v'][:]
    WW = fh.variables['w'][:]
    ref = fh.variables['reflectivity'][:]
    # UU = np.ones_like(ref)*10
    # VV = np.ones_like(ref)*10
    # WW = np.zeros_like(ref)
    fh.close()
    #Get boundaries of NWP and query points
    xmin,xmax = (np.min(XX),np.max(XX))
    ymin,ymax = (np.min(YY),np.max(YY))
    zmin,zmax = (np.min(ZZ),np.max(ZZ))
    xqmin,xqmax = (np.min(xq),np.max(xq))
    yqmin,yqmax = (np.min(yq),np.max(yq))
    zqmin,zqmax = (np.min(zq),np.max(zq))
    
    #Check if all query points fall within NWP volume
    xgood = (xmin<xqmin) & (xmax>xqmax)
    ygood = (ymin<yqmin) & (ymax>yqmax)
    zgood = (zmin<zqmin) & (zmax>zqmax)
    if xgood & ygood & zgood:
        #Load variables from corresponding files and spatially
        #interpolate
        rgi = interpolate.RegularGridInterpolator((XX,YY,ZZ),UU)
        uu = rgi(np.array([xq,yq,zq]).T)
        rgi = interpolate.RegularGridInterpolator((XX,YY,ZZ),VV)
        vv = rgi(np.array([xq,yq,zq]).T)
        rgi = interpolate.RegularGridInterpolator((XX,YY,ZZ),WW)
        ww = rgi(np.array([xq,yq,zq]).T)
        rgi = interpolate.RegularGridInterpolator((XX,YY,ZZ),10**(ref/10))
        ref = 10*np.log10(rgi(np.array([xq,yq,zq]).T))
        return (ref,uu,vv,ww)
    else:
        print('Specified volume out of bounds.')

def getRelPos(sensorPos,peakPt,scatPos):
#Gets relative angular position of some target with respect to the sensor
#pointing angle

#Inputs:
#sensorPos - sensor (Tx/Rx) position
#peakPt - point along pointing direction of sensor
#scatPos - scatterer positions

#Outputs:
#rxtheta - angular distance from boresight
#rxphi - angular rotation about boresight
#rxr - range
    scatPos = scatPos - sensorPos[:,np.newaxis]
    peakPt = peakPt - sensorPos
    
    res = cart2sph(peakPt[0],peakPt[1],peakPt[2])
    az = res[0]
    el = res[1]
    zAngle = -az-np.pi/2
    xAngle = el-np.pi/2
    
    rotZ = rotMat('z',zAngle)
    rotX = rotMat('x',xAngle)
    rotXZ = np.matmul(rotX,rotZ)
    scatPos = np.matmul(rotXZ,scatPos)
    es = np.einsum('ij,ij -> j',scatPos,scatPos)
    rxr = ne.evaluate('sqrt(es)')
    z = scatPos[2,:]/rxr
    rxphi = ne.evaluate('arccos(z)')
    x = scatPos[0,:]
    y = scatPos[1,:]
    rxtheta = ne.evaluate('arctan2(y,x)')
    return (rxtheta, rxphi, rxr)


def cart2sph(x,y,z):
    try:
        n = len(x)
    except:
        n = 1
    ptsnew = np.zeros((n,3))
    xy = ne.evaluate('x**2 + y**2')
    sxy = ne.evaluate('sqrt(xy)')
    ptsnew[:,0] = ne.evaluate('arctan2(y, x)')
    #ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    ptsnew[:,1] = ne.evaluate('arctan2(z, sxy)') # for elevation angle defined from XY-plane up
    ptsnew[:,2] = ne.evaluate('sqrt(xy + z**2)')
    return np.squeeze(ptsnew)

def rotMat(axis,angle):
    R = np.zeros((3,3))
    if axis == 'x':
        R[0,0] = 1;
        R[1,1] = np.cos(angle)
        R[1,2] = -np.sin(angle)
        R[2,1] = np.sin(angle)
        R[2,2] = np.cos(angle)
    elif axis == 'y':
        R[0,0] = np.cos(angle)
        R[0,2] = np.sin(angle)
        R[1,1] = 1
        R[2,0] = -np.sin(angle)
        R[2,2] = np.cos(angle)
            
    elif axis == 'z':
        R[0,0] = np.cos(angle)
        R[0,1] = -np.sin(angle)
        R[1,0] = np.sin(angle)
        R[1,1] = np.cos(angle)
        R[2,2] = 1
    return R

def getUlaWts(theta,phi,rxG):
#Unit tested
#GETULAWTS evaluates antenna pattern for rx antenna (assumed it would be a
# ula) through interpolation from table in rxPat
    uu = np.sin(theta)*np.cos(phi)
    vv = np.sin(theta)*np.sin(phi)
    f = loadmat('rxPat.mat')
    UU = f['UU'][:]
    VV = f['VV'][:]
    rxPat = f['rxPat'][:]
    rxPat = rxPat*10**(rxG/20)/np.max(rxPat)  
    rxPatr = np.real(rxPat)
    rxPati = np.imag(rxPat)
    inter = interpolate.RegularGridInterpolator((UU[0,:],VV[:,0]),rxPatr.T)
    wtsr = inter(np.array([uu,vv]).T)
    inter = interpolate.RegularGridInterpolator((UU[0,:],VV[:,0]),rxPati.T)
    wtsi = inter(np.array([uu,vv]).T)
    wts = wtsr+1j*wtsi
    return wts

def getThetaS(pts,txPos,rxPos):
#Unit tested
#GETTHETAS get forward scattering angle
#
#Inputs
#pts - x,y,z coordinates of scatterers
#txPos - position of transmitter
#rxPos - position of receiver
#
#Outputs:
#thetaS - forward scatter angle

    TS = np.zeros(pts.shape)
    SR = np.zeros(pts.shape)
    
    for ii in range(3):
        TS[ii,:] = pts[ii,:] - txPos[ii]
        SR[ii,:] = rxPos[ii] - pts[ii,:]
    TSmag = np.sqrt(np.sum(TS**2,axis=0))
    SRmag = np.sqrt(np.sum(SR**2,axis=0))
    
    thetaS = np.arccos(np.sum(TS*SR,axis=0)/(TSmag*SRmag))
    return thetaS

def bistaticWeights(txPos,rxPos,pts):
    #Unit tested
    #BISTATICWEIGHTS - gets bistatic scattering weights / loss factors from raindrops assuming dipole
    #model and no roll on antennas (explained in simulator section of Byrd 
    #dissertation)
    #
    #Inputs
    #txPos - transmitter position
    #rxPos - receiver position
    #pts - scatterer locations
    #
    #Outputs
    #Hwts - H polarized scattering weights
    #Vwts - V polarized scattering weights
    rpts = pts.T - txPos
    rrxPos = rxPos.T - txPos

    nPts = pts.shape[1]
    res = cart2sph(rpts[:,0],rpts[:,1],rpts[:,2])
    az,el,r = res[:,0],res[:,1],res[:,2]
    rotZ = np.zeros((nPts,3,3))
    rotY = np.zeros((nPts,3,3))
    rotX = np.zeros((3,3))
    
    rotZ[:,0,0] = np.cos(-az)
    rotZ[:,0,1] = np.sin(-az)
    rotZ[:,1,0] = -np.sin(-az)
    rotZ[:,1,1] = np.cos(-az)
    rotZ[:,2,2] = 1
    
    rotY[:,0,0] = np.cos(el)
    rotY[:,0,2] = np.sin(el)
    rotY[:,1,1] = 1
    rotY[:,2,0] = -np.sin(el)
    rotY[:,2,2] = np.cos(el)
    
    rotX[0,0] = 1
    rotX[1,2] = 1
    rotX[2,1] = -1
    
    res = np.matmul(rotZ,rrxPos[:,np.newaxis])
    vRx = np.matmul(rotY,res)
    vRx[:,0,0] = vRx[:,0,0] - r
    hRx = np.matmul(rotX,vRx)
    
    Vwts = 1-vRx[:,2,:]**2/np.sum(vRx**2,axis=1)
    Hwts = 1-hRx[:,2,:]**2/np.sum(hRx**2,axis=1) 
    return (Hwts, Vwts)

def z2eta(Z,_lambda):
#Z2ETA converts dBZ to eta in m^2/m^3
    Ze = 10**(Z/10)*10**(-18)
    Kw = 0.93
    eta = np.pi**5/_lambda**4*Kw**2*Ze
    return eta

def getBistaticRanges(pos1,pos2,pts):
    #GETBISTATICRANGE gets bistatic ranges for pts w/ respect to pos1 and
    #pos2
    if np.all(pos1 == pos2):
        ranges = 2*np.sqrt(np.sum((pts-pos1[:,np.newaxis])**2,axis=0))
    else:
        ranges = np.sqrt(np.sum((pts-pos1[:,np.newaxis])**2,axis=0)) + np.sqrt(np.sum((pts-pos2[:,np.newaxis])**2,axis=0))
    return ranges

def inRange2(r0,tau,ranges):
    #INRANGE returns mask for points within bistatic range shell
    #centered at r0
    
    c = 3e8
    rNear = r0 - c*tau/2
    rFar  = r0 + c*tau/2
    
    (i1,i2) = findinsorted(ranges,rNear,rFar)
    if i2 == 0:
        return np.array([])
    else:
        shellPts = np.arange(i1,i2+1).astype(int)
        return shellPts

def findinsorted(x,LowerBound,UpperBound):
# fast O(log2(N)) computation of the range of indices of x that satify the
# upper and lower bound values using the fact that the x vector is sorted
# from low to high values. Computation is done via a binary search.
#
# Input:
#
# x-            A vector of sorted values from low to high.       
#
# LowerBound-   Lower boundary on the values of x in the search
#
# UpperBound-   Upper boundary on the values of x in the search
#
# Output:
#
# lower_index-  The smallest index such that
#               LowerBound<=x(index)<=UpperBound
#
# upper_index-  The largest index such that
#               LowerBound<=x(index)<=UpperBound

    if (LowerBound>x[-1]) | (UpperBound<x[0]) | (UpperBound<LowerBound):
        # no indices satisfy bounding conditions
        lower_index = 0;
        upper_index = 0;
        return (lower_index, upper_index)
    lower_index_a=0
    lower_index_b=len(x)-1 # x(lower_index_b) will always satisfy lowerbound
    upper_index_a=0     # x(upper_index_a) will always satisfy upperbound
    upper_index_b=len(x)-1
    
    # The following loop increases _a and decreases _b until they differ 
    # by at most 1. Because one of these index variables always satisfies the 
    # appropriate bound, this means the loop will terminate with either 
    # lower_index_a or lower_index_b having the minimum possible index that 
    # satifies the lower bound, and either upper_index_a or upper_index_b 
    # having the largest possible index that satisfies the upper bound. 

    while (lower_index_a+1<lower_index_b) | (upper_index_a+1<upper_index_b):
    
        lw=np.floor((lower_index_a+lower_index_b)/2) # split the upper index
    
        if x[int(lw)] >= LowerBound:
            lower_index_b=lw # decrease lower_index_b (whose x value remains \geq to lower bound)   
        else:
            lower_index_a=lw # increase lower_index_a (whose x value remains less than lower bound)
            if (lw>upper_index_a) & (lw<upper_index_b):
                upper_index_a=lw # increase upper_index_a (whose x value remains less than lower bound and thus upper bound)

        up=np.ceil((upper_index_a+upper_index_b)/2) # split the lower index
        if x[int(up)] <= UpperBound:
            upper_index_a=up # increase upper_index_a (whose x value remains \leq to upper bound) 
        else:
            upper_index_b=up  # decrease upper_index_b
            if (up<lower_index_b) & (up>lower_index_a):
                lower_index_b=up #decrease lower_index_b (whose x value remains greater than upper bound and thus lower bound)

    
    if x[int(lower_index_a)]>=LowerBound:
        lower_index = lower_index_b
    else:
        lower_index = lower_index_a

    if x[int(upper_index_b)]<=UpperBound:
        upper_index = upper_index_a
    else:
        upper_index = upper_index_b
    return (lower_index, upper_index)

# @njit
# def getArrayWts2(theta,phi,interpolator,codeVal,txG):
#     #Unit tested
#     #GETARRAYWTS@ - gets transmit antenna pattern values at scatterer locations
#     #
#     #Inputs
#     #theta - angular distance from boresight
#     #phi - - angular rotation about boresight
#     #interpolator - set of precalculated patterns
#     #codeVal - whitening code bit
#     #txG - transmitter gain
#     #
#     #Output
#     #wts - transmit antenna pattern weights
#     fftres = 4096
#     st = np.sin(theta)
#     sf = (fftres-1)/2
#     uu = np.empty_like(phi)
#     vv = np.empty_like(phi)
#     uu = np.round((st*np.cos(phi)+1)*sf,0,uu).astype(int64)
#     vv = np.round((st*np.sin(phi)+1)*sf,0,vv).astype(int64)

#     codeVal[codeVal==-1] = 0
#     codeVal = 2*codeVal[1] + codeVal[0]
#     txPat = interpolator[:,:,codeVal]
#     txPat = txPat*10**(txG/20)/np.max(txPat)
#     wts = np.zeros(len(uu),dtype=complex128)
#     for i in range(len(uu)):
#         u = uu[i]
#         v = vv[i]
#         wts[i] = txPat[u,v]
#     return wts




def getArrayWts2(theta,phi,interpolator,codeVal,txG):
    #Unit tested
    #GETARRAYWTS@ - gets transmit antenna pattern values at scatterer locations
    #
    #Inputs
    #theta - angular distance from boresight
    #phi - - angular rotation about boresight
    #interpolator - set of precalculated patterns
    #codeVal - whitening code bit
    #txG - transmitter gain
    #
    #Output
    #wts - transmit antenna pattern weights
    fftres = 4096
    st = ne.evaluate('sin(theta)')
    sf = (fftres-1)/2
    uu = np.round((st*ne.evaluate('cos(phi)')+1)*sf).astype(int)
    vv = np.round((st*ne.evaluate('sin(phi)')+1)*sf).astype(int)
    codeVal[codeVal==-1] = 0
    codeVal = int(2*codeVal[1] + codeVal[0])
    txPat = interpolator[:,:,codeVal]
    idx = np.ravel_multi_index((uu,vv),txPat.shape)
    #wts = txPat[uu,vv]
    wts = txPat.flatten()[idx]
    return wts

def getDishWts(scatTheta,patTheta,txPat):
    inter = interpolate.interp1d(patTheta.flatten()*np.pi/180,txPat.flatten(),kind='nearest',fill_value=0,bounds_error=False)
    wts = inter(scatTheta)
    return wts

def calcV(mski,theta,txWts,curVwts,Pt):
    if len(mski) > 0:
        mskTh = np.abs(theta[mski])<np.pi/2
        rvwt = txWts[mski[mskTh]]*curVwts[mskTh]
        return np.sqrt(Pt)*np.sum(rvwt)
    else:
        return 0
    
def localize(aa,ee,tt,rxPos,txPos):
    c = 3e8
    nRx = rxPos[:,np.newaxis].shape[1]
    if np.all(np.isclose(rxPos,txPos)):
        theta = (90 - ee)*np.pi/180
        phi = aa*np.pi/180
        nAngle = len(theta)
        npts = np.zeros((nAngle,len(tt),3))
        ranges = c*tt/2
        RR,PH = np.meshgrid(ranges,phi)
        r,TH = np.meshgrid(ranges,theta)
        npts[:,:,0]  = r*np.sin(TH)*np.cos(PH)+txPos[0]
        npts[:,:,1]  = r*np.sin(TH)*np.sin(PH)+txPos[1]
        npts[:,:,2]  = r*np.cos(TH)+txPos[2]
    else:
        th = (90 - ee)*np.pi/180
        ph = aa*np.pi/180
        nAngle = len(th)
        pointingVector = np.array([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)])
        #pointingMag = np.sqrt(np.sum(pointingVector**2))
        rxVector = rxPos - np.tile(txPos,nRx)
        rxMag = np.sqrt(np.sum(rxVector**2))
        pp = np.zeros((nRx,nAngle))
        xx = np.zeros((nRx,len(tt),nAngle))
        yy = np.zeros((nRx,len(tt),nAngle))
        zz = np.zeros((nRx,len(tt),nAngle))
        
        npts = np.zeros((nAngle,len(tt),3))
        ca = np.zeros_like(aa).astype(float)
        ce = np.zeros_like(ee).astype(float)
        rxAz = np.arctan2(rxVector[1],rxVector[0])
        rxEl = np.arcsin(rxVector[2]/rxMag)
        D = rxMag
        crx = rotMat('y',rxEl)@rotMat('z',-rxAz)@rxVector[:,np.newaxis]
        for jj in range(nAngle):
            cp = rotMat('y',rxEl)@rotMat('z',-rxAz)@pointingVector[:,jj][:,np.newaxis]
            ca[jj] = np.arctan2(cp[1],cp[0])*180/np.pi
            ce[jj] = np.arcsin(cp[2]/np.sqrt(np.sum(cp**2)))*180/np.pi
            pp[:,jj] = np.arccos(np.sum(crx*cp)/(np.sqrt(np.sum(crx**2))*np.sqrt(np.sum(cp**2))))*180/np.pi
        PP,TT = np.meshgrid(pp,tt)
        AA = np.tile(ca,np.array([len(tt),1]))
        EE = np.tile(ce,np.array([len(tt),1]))
        xx = ((c*TT)**2 - D**2)/(2*(c*TT - D*np.cos(PP*np.pi/180)))*np.cos(AA*np.pi/180)*np.cos(EE*np.pi/180)
        yy = ((c*TT)**2 - D**2)/(2*(c*TT - D*np.cos(PP*np.pi/180)))*np.sin(AA*np.pi/180)*np.cos(EE*np.pi/180)
        zz = ((c*TT)**2 - D**2)/(2*(c*TT - D*np.cos(PP*np.pi/180)))*np.sin(EE*np.pi/180)
        for jj in range(nAngle):
            for kk in range(len(tt)):
                npts[jj,kk,:] =  rotMat('z',rxAz)@rotMat('y',-rxEl)@np.array([xx[kk,jj], yy[kk,jj],zz[kk,jj]])+txPos
    return npts
    
def freq2vel(df,xx,yy,zz,rxPos,txPos,_lambda):
    if np.all(np.isclose(rxPos,txPos)):
        vv = df*_lambda/2
    else:
        txVecs = np.zeros(xx.shape+(3,))
        txVecs[:,:,0] = -xx
        txVecs[:,:,1] = -yy
        txVecs[:,:,2] = -zz
        rxVecs = txVecs + np.tile(np.transpose(rxPos),np.array(xx.shape+(1,)))
        
        txNorm = np.sqrt(np.sum(txVecs**2,axis=2))
        rxNorm = np.sqrt(np.sum(rxVecs**2,axis=2))
        
        beta = np.arccos(np.sum(rxVecs*txVecs,axis=2)/(rxNorm*txNorm))
        vv = _lambda*df/(2*np.cos(beta/2))
    return vv

def getBa(xx,yy,zz,rxPos):
    txVecs = np.zeros(xx.shape + (3,))
    txVecs[:,:,0] = -xx
    txVecs[:,:,1] = -yy
    txVecs[:,:,2] = -zz
    rxVecs = txVecs + np.tile(rxPos[np.newaxis,:],xx.shape+(1,))
    txDist = np.sqrt(np.sum(txVecs**2,axis=2))
    rxDist = np.sqrt(np.sum(rxVecs**2,axis=2))
    beta = np.arccos(np.sum(rxVecs*txVecs,axis=2)/(rxDist*txDist))*180/np.pi
    return beta

def cressman(x1,y1,v1,xq,yq,R,minPts):
    fs = xq.shape
    xq = xq.flatten()
    yq = yq.flatten()
    v1c = np.zeros_like(xq)
    R = R**2
    for jj in range(len(xq)): 
        v1c[jj] = cressmanVal(x1,y1,v1,xq[jj],yq[jj],R,minPts)
    v1c = np.reshape(v1c,fs)
    return v1c

def barnes2(x1,y1,v1,xq,yq,R,minPts):
    fs = xq.shape
    xq = xq.flatten()
    yq = yq.flatten()
    v1c = np.zeros_like(xq)
    R = R**2
    for jj in range(len(xq)): 
        v1c[jj] = barnesVal(x1,y1,v1,xq[jj],yq[jj],R,minPts)
    v1c = np.reshape(v1c,fs)
    return v1c
@njit
def cressmanVal(x1,y1,v1,xq,yq,R,minPts):
    r1 = (x1 - xq)**2 + (y1 - yq)**2
    msk = r1<R
    vt = v1[msk]
    r1 = r1[msk]

    filt = np.isnan(vt)
    if np.sum(filt)/(len(filt)+1e-6) > 0.8 or np.sum(~filt) < minPts:
        return np.nan
    else:
        vt = vt[~filt]
        r1 = r1[~filt]
        w1 = (R - r1)/(R + r1+1e-6)
        s = np.sum(w1)
        if s == 0:
            return np.nan
        else:
            return np.sum(w1*vt)/np.sum(w1)
        
@njit
def barnesVal(x1,y1,v1,xq,yq,R,minPts):
    r1 = (x1 - xq)**2 + (y1 - yq)**2
    msk = r1<R
    vt = v1[msk]
    r1 = r1[msk]

    filt = np.isnan(vt)
    if np.sum(filt)/(len(filt)+1e-6) > 0.8 or np.sum(~filt) < minPts:
        return np.nan
    else:
        vt = vt[~filt]
        r1 = r1[~filt]
        w1 = np.exp(-r1/(R/4))+1e-5
        s = np.sum(w1)
        if s == 0:
            return np.nan
        else:
            return np.sum(w1*vt)/np.sum(w1)

def dealias(vr):
    nr,nb = vr.shape
    vr_d = np.copy(vr)
    va = np.nanmax(vr)
    adj = np.array([0, 2*va, -2*va, 4*va, -4*va])
    # Find lowest non-NaN value in first range ring (least likely to be aliased)
    rad1 = np.where(abs(vr_d[:,0])==np.nanmin(abs(vr_d[:,0])))[0][0]
    # Fill in NaN values in first range ring
    for j in np.arange(rad1, nr):
        if np.isnan(vr_d[j,0]) & np.isfinite(vr_d[j-1,0]):
            if j == 0:
                vr_d[j,0] = vr_d[nr-1,0]
            else:
                vr_d[j,0] = vr_d[j-1,0]
    for j in np.arange(0,rad1):
        if j == 0:
            if np.isnan(vr_d[j,0]) & np.isfinite(vr_d[nr-1,0]):
                vr_d[j,0] = vr_d[nr-1,0]
        else:
            if np.isnan(vr_d[j,0]) & np.isfinite(vr_d[j-1,0]):
                vr_d[j,0] = vr_d[j-1,0]
    # Radially fill all NaN values
    for i in np.arange(1,nb):
        j = np.isnan(vr_d[:,i]) & np.isfinite(vr_d[:,i-1])
        vr_d[j,i] = vr_d[j,i-1]
    nrad = 3
    nbin = 2
    window = np.ones((nrad,nbin))
    for j in np.arange(2,nb):
        vals = vr_d[:,j]+adj[:,np.newaxis]
        m = signal.convolve2d(vr_d[:,j-nbin:j],window,boundary='wrap')[nrad//2:-(nrad//2),0]/(nrad*nbin)
        pos = abs(vals-m).argmin(axis=0)
        vr_d[:,j] += adj[pos]
        
    # 1st radial dealiasing pass
    
        
    # Find least aliased radial
    r = np.mean(abs(vr_d),axis=1).argmin()
    #Azimuthal deliasing pass
    for i in np.arange(r+1,nr):
        vals = vr_d[i,:]+adj[:,np.newaxis]
        pos = abs(vr_d[i-1,:]-vals).argmin(axis=0)
        vr_d[i,:] += adj[pos]
    for i in np.arange(0, r-1):
        vals = vr_d[i,:]+adj[:,np.newaxis]
        if i == 0:
            pos = abs(vr_d[nr-1,:]-vals).argmin(axis=0)
        else:
            pos = abs(vr_d[i-1,:]-vals).argmin(axis=0)
        vr_d[i,:] += adj[pos]
    # 2nd radial dealiasing pass with higher constraints
    for j in np.arange(2,nb):
        vals = vr_d[:,j]+adj[:,np.newaxis]
        pos = abs(vals-np.mean(vr_d[:,j-2:j],axis=1)).argmin(axis=0)
        vr_d[:,j] += adj[pos]
    return vr_d

def simulate(radarStruct,wxStruct):
    os.chdir('/Users/semmerson/Documents/MATLAB/bistaticSimulator')
    f = loadmat('gold64.mat')
    c1 = np.squeeze(f['c1'][:])
    c2 = np.squeeze(f['c2'][:])
    c1 = np.ones(c1.shape)
    c2 = np.ones(c1.shape)
    txPos = radarStruct['txPos'] #Transmitter pos
    rxPos = radarStruct['rxPos'] #Receiver pos
    _lambda = radarStruct['lambda']#Operating wavelength
    prt = radarStruct['prt'] #Pulse Repetition Time
    tau = radarStruct['tau'] #pulse width
    fs = radarStruct['fs'] #sampling rate
    Pt = radarStruct['Pt'] #transmit power
    Tant = radarStruct['Tant'] #Antenna noise/brightness temp
    Frx = radarStruct['Frx'] #Receiver noise figure
    M = radarStruct['M'] #Samples per dwell
    txAz = radarStruct['txAz'] #Transmit azimuths
    txEl = radarStruct['txEl'] #Transmit elevations
    rxMechTilt = radarStruct['rxMechTilt'] #Mechanical tilt of rx antenna
    txMechEl = radarStruct['txMechEl'] #Mechanical tilt of tx antenna
    txG = radarStruct['txG'] #Transmit antenna gain (dBI)
    rxG = radarStruct['rxG'] #Receive antenna gain
    
    scatMode = wxStruct['scatMode'] #Wx or Bragg scatter
    xSize = wxStruct['xSize'] #Sizes of each dimension of Wx volume
    ySize = wxStruct['ySize']
    zSize = wxStruct['zSize']
    wrfDate = wxStruct['wrfDate'] #Date to be queried from WRF data
    
    #Spatial offset of sim volume within WRF volume
    wrfOffset = wxStruct['wrfOffset']
    
    #minimum number of scatterers in smallest volume
    ptsPerMinVol = wxStruct['ptsPerMinVol']    
    
    f = loadmat('win88d.mat')
    baseWin = np.squeeze(f['baseWin'][:])
    f = loadmat('win88dWht5.mat')
    codingWeights1 = np.squeeze(f['totalWts1'][:])
    codingWeights2 = np.squeeze(f['totalWts2'][:])
    codingWeights1 = baseWin
    codingWeights2 = baseWin
    idealPat = False
    dishMode = True
    fftres = 4096;
    wts = np.zeros((len(codingWeights1),len(codingWeights1),4)).astype('complex128')
    wts[:,:,0] = codingWeights1*codingWeights1[:,np.newaxis]
    wts[:,:,1] = codingWeights1*codingWeights2[:,np.newaxis]
    wts[:,:,2] = codingWeights2*codingWeights1[:,np.newaxis]
    wts[:,:,3]= codingWeights2*codingWeights2[:,np.newaxis]
    patMatrix = np.zeros((fftres,fftres,4)).astype('complex128')
    
    for ii in range(4):
        txPat = fftshift(fft2(wts[:,:,ii],shape = (fftres,fftres)))
        if idealPat: #Zero all sidelobes in ideal case
            maxi = round((0+1)/2*(fftres-1))+1
            maxj = round((-0+1)/2*(fftres-1))+1
            uslice = txPat[maxi,maxj:]
            vslice = txPat[maxi:,maxj]
            negIndsu = np.where((np.diff(np.abs(uslice))>0) & (np.abs(uslice[1:])<(0.5*np.max(np.abs(uslice[1:])))))[0]
            firstNullu = negIndsu[0]
            negIndsv = np.where((np.diff(np.abs(vslice))>0) & (np.abs(vslice[1:])<(0.5*np.max(np.abs(vslice[1:])))))[0]
            firstNullv = negIndsv[0]
            iu = maxi + firstNullv - 1
            il = maxi - firstNullv
            ju = maxj + firstNullu - 1
            jl = maxj - firstNullu
            txPat[:il,:] = 0
            txPat[iu:,:] = 0
            txPat[:,:jl] = 0
            txPat[:,ju:] = 0
        txPat = txPat*10**(txG/20)/np.max(txPat)
        patMatrix[:,:,ii] = txPat
    c = 3e8 #speed of light
    nRx = rxPos.shape[0] ## of Rx
    Ts = 1/fs #sampling interval
    T0 = 290 #reference noise temp
    Trec = T0*(10**(Frx/10)-1) #Rx noise temp
    B = fs #receiver bandwidth (assumes IQ sampling)
    k = 1.381e-23 #Boltzmann
    N0 = k*(Tant+Trec)*B #Noise power at receiver
    cn = 1e-14
    
    # Populate scattering centers
    domainVolume = xSize*ySize*zSize
    minVolSize = (np.sin(0.5*np.pi/180)*35e3)**2*c*tau/2
    nPts = np.round(ptsPerMinVol * domainVolume/minVolSize)
    ptVolume = domainVolume/nPts
    
    np.random.seed(777);
    pts = np.zeros((3,int(nPts)))
    pts[0,:] = xSize*np.random.uniform(size=(int(nPts))) - xSize/2
    pts[1,:] = ySize*np.random.uniform(size=(int(nPts))) - ySize/2
    pts[2,:] = zSize*np.random.uniform(size=(int(nPts)))
    # pts[0,:] = xSize*np.arange(0.1,1,0.1) - xSize/2
    # pts[1,:] = ySize*np.arange(0.1,1,0.1) - ySize/2
    # pts[2,:] = zSize*np.arange(0.1,1,0.1)
    os.chdir('/Users/semmerson/Documents/cm1r19.10/grinder')
    #os.chdir('/Users/semmerson/Documents/python/data/may7/may_7_250m')
    #Zv,uu,vv,ww = getWrf(wrfDate,pts[0,:]/1000+wrfOffset[0],pts[1,:]/1000+wrfOffset[1],pts[2,:]/1000+wrfOffset[2])
    Zv,uu,vv,ww = getWrf(pts[0,:]/1000+wrfOffset[0],pts[1,:]/1000+wrfOffset[1],pts[2,:]/1000+wrfOffset[2])
    os.chdir('/Users/semmerson/Documents/MATLAB/bistaticSimulator')
    #Vectorize wind velocities
    windSpeed = np.zeros((3,int(nPts)))
    windSpeed[0,:] = uu
    windSpeed[1,:] = vv
    windSpeed[2,:] = ww
    
    del uu,vv,ww
    
    
    # xc = -10
    # yc = 0
    # window = 10
    # x = (xc-window, xc+window)
    # y = (yc-window, yc+window)
    # def animate(t,pts,windSpeed):
    #     plt.clf()
    #     pts = pts+0.5*t*windSpeed
    #     ax = fig.add_subplot(111)
    #     ax.scatter(pts[0,:50000]/1000,pts[1,:50000]/1000,c=Zv[:50000])
    #     ax.set_xlim(x)
    #     ax.set_ylim(y)
    #     plt.tight_layout()
    # fig = plt.figure(figsize=(8, 6))
    # anim = mani.FuncAnimation(fig, animate, interval=1, frames=60,fargs=(pts,windSpeed))
    # anim.save('test.gif',writer='imagemagick',fps=30)
    
    
    staticVwts = np.zeros((int(nRx),int(nPts))).astype('complex128')
    print('Calculating Bistatic RCS:')
    rxr = np.sqrt(np.sum(rxPos**2,axis=1))
    rxz = rxr*np.sin(rxMechTilt*np.pi/180)
    
    rxRef = np.vstack((np.zeros((2,int(nRx))), rxz))
    
    txPhi = (90-txAz)*np.pi/180;
    txMechTheta = (90-txMechEl)*np.pi/180;
    nAz = txAz.shape[0]
    nEl = txMechTheta.shape[0]
    
    Vpower = []
    rr = []
    nr = np.zeros((nRx),dtype=int)
    for i in range(nRx):
        brmin = np.sqrt(np.sum((rxPos[i,:]-txPos)**2))
        brmax = 1e5
        # if np.all(np.isclose(rxPos[i,:],txPos)):
        #     brmax = brmax/2
        rr.append(np.arange(brmin,brmax,c*Ts/2))
        nr[i] = len(rr[i])
        Vpower.append(np.zeros((nAz,nEl,nr[i],M)).astype('complex128'))
    
    #Calculate portion of complex scatterer weights that will not change
    #with Tx pointing angle.
    #
    #Makes simplifying assumption that Tx scan happens instantaneously
    if dishMode:
        f = loadmat('dishWts.mat')
        patTheta = f['patTheta'][:]
        txPat = f['txPat'][:]
    for qq in range(M): #For samples in dwell
        pts = pts + prt*windSpeed
        br = np.zeros(staticVwts.shape)
        #Reference pointing direction vector for transmitter

        #Initialize variables
        nPts = pts.shape[1]
        rxPhi = np.zeros((nRx,nPts))
        rxTheta = np.zeros((nRx,nPts))
        rxr = np.zeros((nRx,nPts))
        for ii in range(nRx): #For each receiver          
            #Find effective receive ant area in direction of scatterers
            print('Receiver ',ii+1,' of ',nRx)
    
            #Get scatterer positions relative to receiver
            (rxTheta[ii,:], rxPhi[ii,:], rxr[ii,:]) = getRelPos(rxPos[ii,:],rxRef[:,ii],np.squeeze(pts))
            #Get Rx antenna pattern weightings 
            Ae = getUlaWts(rxTheta[ii,:],rxPhi[ii,:],rxG)*_lambda**2/(4*np.pi)
    
            #Get forward scatter angles
            #thetaS = getThetaS(pts,txPos,rxPos[ii,:])
            #Get dual-pol scattering amplitudes based on infinitesimal
            #dipole-like reradiation
            vwt = np.squeeze(bistaticWeights(txPos,rxPos[ii,:],pts)[1])
    
            # Do the "receive" half of the radar range equation (doesn't
            # change with Tx pointing angle.  Convert from Z or Cn^2 to
            # reflectivity (eta) as necessary
            if scatMode == 'rayleigh':
                staticVwts[ii,:] = np.sqrt(vwt*ptVolume*z2eta(Zv,0.1)*Ae)/(4*np.pi*rxr[ii,:])*np.exp(1j*2*np.pi*rxr[ii,:]/_lambda)
            elif scatMode == 'bragg':
                print('bragg?')
                #staticVwts[ii,:] = np.sqrt(vwt*ptVolume*cn2eta(_lambda,cn,thetaS)*Ae)/(4*np.pi*rxr[ii,:,:])
            # Get total distance traveled from tx to each point back to rx
            br[ii,:] = getBistaticRanges(txPos,rxPos[ii,:],pts)
        #Format bistatic range info 
        br = np.squeeze(br)
        if nRx == 1:
            br = br.T
            
        # #Filter out unused scatterers based on bistatic range
        rmsk = np.min(br,axis=0)>brmax
        br = br[:,~rmsk]
        staticVwts = staticVwts[:,~rmsk]
        pts = pts[:,~rmsk]
        windSpeed = windSpeed[:,~rmsk]
        Zv = Zv[~rmsk]
        rxr = rxr[:,~rmsk]
            
        sortInds = np.zeros(br.shape).astype(int)
        for ii in range(nRx):
            sortInds[ii,:] = np.argsort(br[ii,:])
            br[ii,:] = np.sort(br[ii,:])
            staticVwts[ii,:] = staticVwts[ii,sortInds[ii,:]]
        staticVwts[np.isnan(staticVwts)] = 0
        del Ae,rxPhi,rxTheta 
        print('Simulating Observations:')

        for jj in range(nRx): #for each receiver
              #Get sorted pts by bistatic range for current receiver
              ptsr = pts[:,sortInds[jj,:]]

              #Initialize cell arrays for storing range bin data
              ncp = np.zeros((nr[jj]))
              mskIndices = []
              curVwts = []
              for ll in range(nEl):
                  txRef = np.vstack((np.sin(txMechTheta[ll])*np.cos(txPhi)+txPos[0],
                      np.sin(txMechTheta[ll])*np.sin(txPhi)+txPos[1],
                      np.cos(txMechTheta[ll])*np.ones(txPhi.shape)+txPos[2]))
                  for kk in range(nr[jj]):
                      mski = inRange2(rr[jj][kk],tau,br[jj,:]) #Find points in bin
                      mskIndices.append(mski)
                      ncp[kk] = len(mski)
                      if len(mski) > 0:
                          curVwts.append(staticVwts[jj,mski]) #Get wts for those points
                      else:
                          curVwts.append(np.array([0]))
                  for ii in range(nAz): #For each azimuth angle
                      if np.mod(ii+1,5) == 0: #Progress display
                          print('Pt', qq+1, 'Rx', jj+1, ': Elevation',ll+1,'/',nEl,'Azimuth',ii+1,'/',nAz)
         
                      #Get position relative to receiver
                      (phi, theta,r) = getRelPos(txPos,txRef[:,ii],ptsr)
                      #Calculate terms for phase delay and attenuation
                      rf = -1j*2*np.pi*r/_lambda
                      rangeFactor = r*ne.evaluate('exp(rf)') 
                      #Get transmit antenna weights
                      if dishMode:
                          txWts = getDishWts(theta,patTheta,txPat)/rangeFactor
                      else:
                          txWts = getArrayWts2(theta,phi,patMatrix,np.array((c1[qq], c2[qq])),txG)/rangeFactor
         
                      #Sum across samples and scale to form IQ
                      for kk in range(nr[jj]):
                          Vpower[jj][ii,ll,kk,qq] = calcV(mskIndices[kk],theta,txWts,curVwts[kk],Pt)
        
    return Vpower, rr
# lp = LineProfiler()
# lp_wrap = lp(simulate)
# lp_wrap()
# lp.print_stats()
    
radarStruct = makeRadarStruct()
wxStruct = makeWxStruct()
Vpower, rr = simulate(radarStruct,wxStruct)

txPos = radarStruct['txPos']
rxPos = radarStruct['rxPos']
_lambda = radarStruct['lambda']
prt = radarStruct['prt']
tau = radarStruct['tau']
fs = radarStruct['fs'] 
Pt = radarStruct['Pt'] 
Tant = radarStruct['Tant'] 
Frx = radarStruct['Frx'] 
M = radarStruct['M'] 
txAz = radarStruct['txAz'] 
txEl = radarStruct['txEl'] 
rxMechTilt = radarStruct['rxMechTilt'] 
txMechEl = radarStruct['txMechEl'] 
txG = radarStruct['txG']
rxG = radarStruct['rxG'] 

scatMode = wxStruct['scatMode'] 
xSize = wxStruct['xSize'] 
ySize = wxStruct['ySize']
zSize = wxStruct['zSize']
wrfDate = wxStruct['wrfDate'] 
wrfOffset = wxStruct['wrfOffset']

c = 3e8 
nRx = rxPos.shape[0]
nEl = txMechEl.shape[0]
Ts = 1/fs 
T0 = 290
Trec = T0*(10**(Frx/10)-1) 
B = fs 
k = 1.381e-23 
N0 = k*(Tant+Trec)*B 
cn = 1e-14

f = loadmat('wrfDirectory.mat') #load NWP directory
XX = f['XX'][:]-wrfOffset[0]
YY = f['YY'][:]-wrfOffset[1]
ZZ = f['ZZ'][:]

saveMode = True

vpp = []
snr = []
df = []
vels = []
xx = []
yy = []
zz = []
br = []
beta = []
snrThresh = 10**(20/10)
brThresh = 1e3

txrxDist = np.sqrt(np.sum((txPos-rxPos)**2,axis=1))
rxPhi = np.arctan2(rxPos[:,1]-txPos[1],rxPos[:,0]-txPos[0])

rxPos2 = np.array([np.cos(rxPhi), np.sin(rxPhi), np.zeros(len(txrxDist))])*txrxDist
txPos2 = np.zeros(3)
for i in range(nRx):
    noiseMat = np.sqrt(N0)/np.sqrt(2)*(np.random.normal(0,1,Vpower[i].shape) + 1j*np.random.normal(0,1,Vpower[i].shape))
    Vpower[i] += noiseMat
    vpp.append(np.sum(np.abs(Vpower[i])**2,axis=3)/radarStruct['M'])
    snr.append(vpp[i]/N0)
    f = np.angle(np.sum(np.conj(Vpower[i][:,:,:,:-1])*Vpower[i][:,:,:,1:],axis=3))
    vpp[i][snr[i] < snrThresh] = np.nan
    f[snr[i] < snrThresh] = np.nan
    df.append(f/(2*np.pi*radarStruct['prt']))
    # br.append(np.reshape(getBistaticRanges(txPos2,rxPos2[:,i],np.array((xx[i].flatten(),yy[i].flatten(),zz[i].flatten()))),xx[i].shape))
    # beta.append(getBa(xx[i],yy[i],zz[i],rxPos2[:,i]))
    vel = np.zeros(df[i].shape)
    xx0 = np.zeros(df[i].shape)
    yy0 = np.zeros(df[i].shape)
    zz0 = np.zeros(df[i].shape)
    for l in range(nEl):
        pts = localize(90-txAz,radarStruct['txMechEl'][l]*np.ones(txAz.shape),rr[i]/c,rxPos2[:,i],txPos2)
        #pts = localize(90-txAz,radarStruct['txMechEl'][l]*np.ones(txAz.shape),br[i][0,:]/c,rxPos2[:,i],txPos2)
        vel[:,l,:] = freq2vel(df[i][:,l,:],pts[:,:,0],pts[:,:,1],pts[:,:,2],rxPos2[:,i],txPos2,_lambda)
        xx0[:,l,:] = pts[:,:,0]
        yy0[:,l,:] = pts[:,:,1]
        zz0[:,l,:] = pts[:,:,2]
    vels.append(vel)
    xx.append(xx0)
    yy.append(yy0)
    zz.append(zz0)
#RR = np.sqrt(np.sum(pts**2,axis=2))
#br = np.reshape(getBistaticRanges(txPos2,rxPos2[:,rxN],np.array((xx0.flatten(),yy0.flatten(),zz0.flatten()))),xx0.shape)
#vels0 = freq2vel(df[rxN],xx0,yy0,zz0,rxPos2[:,rxN],txPos2,_lambda)
#vels0[snr[rxN,:,:] < snrThresh] = np.nan
# vpp[rxN,snr[rxN,:,:] < snrThresh] = np.nan
#vpp[rxN,br < brThresh] = np.nan
#vels0[br < brThresh] = np.nan
xc = 0
yc = 12
window = 15
x = (xc-window, xc+window)
y = (yc-window, yc+window)
l = 0
figsize = (14,8)
if nRx > 5:
    fig = plt.figure(figsize=figsize)
    m = int(np.ceil(np.sqrt(nRx)))-1
    n = int(np.ceil(nRx/m))-1
    axs = fig.subplots(m,n).flatten()
    for i in range(nRx-1):
        pc = axs[i].pcolormesh(xx[i][:,l,:]/1000,yy[i][:,l,:]/1000,vels[i][:,l,:],vmin=-40,vmax=40,cmap='pyart_balance',shading='auto')
        #ax.quiver(XX[::2,::2,1],YY[::2,::2,1],UU[::2,::2,1],VV[::2,::2,1],scale = 600)
        axs[i].set_xlim(x)
        axs[i].set_ylim(y)
        axs[i].scatter(x=0,y=0,marker='x',color='k',s=100)
        axs[i].scatter(x=rxPos2[0,i]/1000,y=rxPos2[1,i]/1000,marker='+',color='r',s=100)
        if i == nRx-1:
            axs[i].set_title('Tx Simulated Radial Velocity')
        else:
            axs[i].set_title('Rx '+ str(i+1) +' Simulated Bistatic Velocity')
        axs[i].set_xlabel('Zonal Distance from Tx Radar (km)')
        axs[i].set_ylabel('Meridional Distance from Tx Radar (km)')
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.05, 0.02, 0.92])
    cb = fig.colorbar(pc, cax=cbar_ax)
    cb.set_label('Bistatic Velocity (m/s)')
    # for ax in fig.get_axes():
    #     ax.label_outer()

    
    fig = plt.figure(figsize=figsize)
    axs = fig.subplots(m,n).flatten()
    for i in range(nRx-1):
        rFact = (rxPos2[0,i]-xx[i][:,l,:])**2+(rxPos2[1,i]-yy[i][:,l,:])**2++(rxPos2[2,i]-zz[i][:,l,:])**2
        #pc = axs[i].pcolormesh(xx[i][:,l,:]/1000,yy[i][:,l,:]/1000,10*np.log10(vpp[i][:,l,:]*rFact),vmin=-70,vmax=10,cmap='pyart_HomeyerRainbow')
        pc = axs[i].pcolormesh(xx[i][:,l,:]/1000,yy[i][:,l,:]/1000,10*np.log10(snr[i][:,l,:]),vmin=-10,vmax=60,cmap='pyart_HomeyerRainbow')
        axs[i].set_xlim(x)
        axs[i].set_ylim(y)
        axs[i].scatter(x=0,y=0,marker='x',color='k',s=100)
        axs[i].scatter(x=rxPos2[0,i]/1000,y=rxPos2[1,i]/1000,marker='+',color='r',s=100)
        if i == nRx-1:
            axs[i].set_title('Tx Simulated Range-Corrected Power')
        else:
            axs[i].set_title('Rx '+ str(i+1) +' Simulated Range-Corrected Power')
        axs[i].set_xlabel('Zonal Distance from Tx Radar (km)')
        axs[i].set_ylabel('Meridional Distance from Tx Radar (km)')
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.05, 0.02, 0.92])
    cb = fig.colorbar(pc, cax=cbar_ax)
    cb.set_label('Range-Corrected Power (dB)')
    # for ax in fig.get_axes():
    #     ax.label_outer()


else:
    fig = plt.figure(figsize=figsize)
    axs = fig.subplots(2,nRx)
    for i in range(nRx):
        pc = axs[0,i].pcolormesh(xx[i][:,l,:]/1000,yy[i][:,l,:]/1000,vels[i][:,l,:],vmin=-40,vmax=40,cmap='pyart_balance',shading='auto')
        #ax.quiver(XX[::2,::2,1],YY[::2,::2,1],UU[::2,::2,1],VV[::2,::2,1],scale = 600)
        axs[0,i].set_xlim(x)
        axs[0,i].set_ylim(y)
        axs[0,i].scatter(x=0,y=0,marker='x',s=100)
        axs[0,i].scatter(x=rxPos2[0,i]/1000,y=rxPos2[1,i]/1000,marker='+',s=100)
        if i == nRx-1:
            axs[0,i].set_title('Tx Simulated Radial Velocity')
            cb = plt.colorbar(pc,ax=axs[0,i])
            cb.set_label('Bistatic Velocity (m/s)')
        else:
            axs[0,i].set_title('Rx '+ str(i+1) +' Simulated Bistatic Velocity')
        axs[0,i].set_xlabel('Zonal Distance from Tx Radar (km)')
        axs[0,i].set_ylabel('Meridional Distance from Tx Radar (km)')
    
        rFact = (rxPos2[1,i]-yy[i][:,l,:])**2+(rxPos2[0,i]-xx[i][:,l,:])**2
        pc = axs[1,i].pcolormesh(xx[i][:,l,:]/1000,yy[i][:,l,:]/1000,10*np.log10(vpp[i][:,l,:]*rFact),vmin=-70,vmax=10,cmap='pyart_HomeyerRainbow')
        axs[1,i].scatter(x=0,y=0,marker='x',s=100)
        axs[1,i].scatter(x=rxPos2[0,i]/1000,y=rxPos2[1,i]/1000,marker='+',s=100)
        if i == nRx-1:
            axs[1,i].set_title('Tx Simulated Range-Corrected Power')
            cb = plt.colorbar(pc,ax=axs[1,i])
            cb.set_label('Range-Corrected Power (dB)')
        else:
            axs[1,i].set_title('Rx '+ str(i+1) +' Simulated Range-Corrected Power')
        axs[1,i].set_xlabel('Zonal Distance from Tx Radar (km)')
        axs[1,i].set_ylabel('Meridional Distance from Tx Radar (km)')
        axs[1,i].set_xlim(x)
        axs[1,i].set_ylim(y)
    for ax in fig.get_axes():
        ax.label_outer()
    plt.tight_layout()
# rxN = 1
# ax = fig.add_subplot(232)
# pc = ax.pcolormesh(xx[rxN]/1000,yy[rxN]/1000,vels[rxN],vmin=-40,vmax=40,cmap='pyart_balance',shading='auto')
# #ax.quiver(XX[::2,::2,1],YY[::2,::2,1],UU[::2,::2,1],VV[::2,::2,1],scale = 600)
# ax.set_xlim(x)
# ax.set_ylim(y)
# ax.scatter(x=0,y=0,marker='x',s=100)
# ax.scatter(x=rxPos2[0,rxN]/1000,y=rxPos2[1,rxN]/1000,marker='+',s=100)
# ax.set_title('Rx 2 Simulated Bistatic Velocity')
# ax.set_xlabel('Zonal Distance from Tx Radar (km)')
# ax.set_ylabel('Meridional Distance from Tx Radar (km)')
# cb = plt.colorbar(pc,ax=ax)
# cb.set_label('Bistatic Velocity (m/s)')
# ax = fig.add_subplot(235)
# rFact = (rxPos2[1,rxN]-yy[rxN])**2+(rxPos2[0,rxN]-xx[rxN])**2
# pc = ax.pcolormesh(xx[rxN]/1000,yy[rxN]/1000,10*np.log10(vpp[rxN]*rFact),vmin=-70,vmax=10,cmap='pyart_HomeyerRainbow')
# cb = plt.colorbar(pc,ax=ax)
# cb.set_label('Range-Corrected Power (dB)')
# ax.set_title('Rx 2 Simulated Range-Corrected Power')
# ax.set_xlabel('Zonal Distance from Tx Radar (km)')
# ax.set_ylabel('Meridional Distance from Tx Radar (km)')
# ax.set_xlim(x)
# ax.set_ylim(y)

# rxN = 2
# ax = fig.add_subplot(233)
# pc = ax.pcolormesh(xx[rxN]/1000,yy[rxN]/1000,vels[rxN],vmin=-40,vmax=40,cmap='pyart_balance',shading='auto')
# #ax.quiver(XX[::2,::2,1],YY[::2,::2,1],UU[::2,::2,1],VV[::2,::2,1],scale = 600)
# ax.set_xlim(x)
# ax.set_ylim(y)
# ax.scatter(x=0,y=0,marker='x',s=100)
# ax.scatter(x=rxPos2[0,rxN]/1000,y=rxPos2[1,rxN]/1000,marker='+',s=100)
# ax.set_title('Tx Simulated Radial Velocity')
# ax.set_xlabel('Zonal Distance from Tx Radar (km)')
# ax.set_ylabel('Meridional Distance from Tx Radar (km)')
# cb = plt.colorbar(pc,ax=ax)
# cb.set_label('Bistatic Velocity (m/s)')
# ax = fig.add_subplot(236)
# rFact = (rxPos2[1,rxN]-yy[rxN])**2+(rxPos2[0,rxN]-xx[rxN])**2
# pc = ax.pcolormesh(xx[rxN]/1000,yy[rxN]/1000,10*np.log10(vpp[rxN]*rFact),vmin=-70,vmax=10,cmap='pyart_HomeyerRainbow')
# cb = plt.colorbar(pc,ax=ax)
# cb.set_label('Range-Corrected Power (dB)')
# ax.set_title('Tx Simulated Range-Corrected Power')
# ax.set_xlabel('Zonal Distance from Tx Radar (km)')
# ax.set_ylabel('Meridional Distance from Tx Radar (km)')
# ax.set_xlim(x)
# ax.set_ylim(y)
# plt.tight_layout()
snrt = 3
#

rilThresh = 10*1e3
hscThresh = 10*1e3
betaThresh = [20,160]

# rPI = np.reshape(interpolate.griddata(np.vstack((xx0.flatten(),yy0.flatten())).T,vpp[0].flatten(),np.vstack((xx1.flatten(),yy1.flatten())).T,method='nearest'),xx1.shape)
# hPI = np.reshape(interpolate.griddata(np.vstack((xx1.flatten(),yy1.flatten())).T,vpp[1].flatten(),np.vstack((xx0.flatten(),yy0.flatten())).T,method='nearest'),xx0.shape)
# rBI = np.reshape(interpolate.griddata(np.vstack((xx0.flatten(),yy0.flatten())).T,brRil.flatten(),np.vstack((xx1.flatten(),yy1.flatten())).T,method='nearest'),xx1.shape)
# hBI = np.reshape(interpolate.griddata(np.vstack((xx1.flatten(),yy1.flatten())).T,brHsc.flatten(),np.vstack((xx0.flatten(),yy0.flatten())).T,method='nearest'),xx0.shape)
# rBeI = np.reshape(interpolate.griddata(np.vstack((xx0.flatten(),yy0.flatten())).T,betaRil.flatten(),np.vstack((xx1.flatten(),yy1.flatten())).T,method='nearest'),xx1.shape)
# hBeI = np.reshape(interpolate.griddata(np.vstack((xx1.flatten(),yy1.flatten())).T,betaHsc.flatten(),np.vstack((xx0.flatten(),yy0.flatten())).T,method='nearest'),xx0.shape)
# tvh = np.nanmin(np.stack((10*np.log10(rPI),vpp[1])),axis=0)
# tvr = np.nanmin(np.stack((10*np.log10(hPI),vpp[0])),axis=0)
# #hscMsk = np.any(np.stack((tvh < 7,np.isnan(tvh),brHsc < hscThresh,rBI < rilThresh, hsc['yy'] > hscPos[1] + 1)),axis=0)
# #rilMsk = np.any(np.stack((tvr < 7,np.isnan(tvr),brRil < rilThresh,hBI < hscThresh, ril['yy'] < rilPos[1] - 1)),axis=0)
# hscMsk = np.any(np.stack((tvh < -160,np.isnan(tvh),brHsc < hscThresh,rBI < rilThresh,rBeI < betaThresh[0], rBeI > betaThresh[1])),axis=0)
# rilMsk = np.any(np.stack((tvr < -160,np.isnan(tvr),brRil < rilThresh,hBI < hscThresh,hBeI < betaThresh[0], hBeI > betaThresh[1])),axis=0)

# nmsk = np.isnan(vels[0])
# vels[0][nmsk] = np.inf*(-1*np.ones((np.sum(nmsk))))**np.round(np.random.uniform(size=np.sum(nmsk)))
# #vels0 = ndimage.median_filter(vels0,footprint = np.ones((3,5)),mode='constant')
# vels[0][np.isinf(vels[0])] = np.nan

# nmsk = np.isnan(vpp[0])
# vpp[0][nmsk] = np.inf*(-1*np.ones((np.sum(nmsk))))**np.round(np.random.uniform(size=np.sum(nmsk)))
# #vpp[0,:,:] = ndimage.median_filter(vpp[0,:,:],footprint = np.ones((3,5)),mode='constant')
# vpp[0][np.isinf(vpp[0])] = np.nan

# nmsk = np.isnan(vels[1])
# vels[1][nmsk] = np.inf*(-1*np.ones((np.sum(nmsk))))**np.round(np.random.uniform(size=np.sum(nmsk)))
# #vels1 = ndimage.median_filter(vels1,footprint = np.ones((3,5)),mode='constant')
# vels[1][np.isinf(vels1)] = np.nan

# nmsk = np.isnan(vpp[1])
# vpp[1][nmsk] = np.inf*(-1*np.ones((np.sum(nmsk))))**np.round(np.random.uniform(size=np.sum(nmsk)))
# #vpp[1,:,:] = ndimage.median_filter(vpp[1,:,:],footprint = np.ones((3,5)),mode='constant')
# vpp[1][np.isinf(vpp[1])] = np.nan

# vels[0][rilMsk] = np.nan
# vpp[0][rilMsk] = np.nan
# vels[1][hscMsk] = np.nan
# vpp[1][hscMsk] = np.nan

rxc = 0e3
ryc = 15e3
rwindow = 15*1e3
roiLims = [rxc-rwindow ,rxc+rwindow, ryc-rwindow, ryc+rwindow]

margin = 1
R = 0.4e3
res = 0.125e3
a = 6.371e6
xq = np.linspace(roiLims[0],roiLims[1],int((roiLims[1]-roiLims[0])/res)+1)
yq = np.linspace(roiLims[2],roiLims[3],int((roiLims[3]-roiLims[2])/res)+1)
xq,yq = np.meshgrid(xq,yq)
r = np.sqrt(xq**2+yq**2)
us = np.zeros((nEl,xq.shape[0],xq.shape[1]))
vs = np.zeros((nEl,xq.shape[0],xq.shape[1]))
ws = np.zeros((nEl,xq.shape[0],xq.shape[1]))
zqs = np.zeros((nEl,xq.shape[0],xq.shape[1]))
for l in range(len(radarStruct['txMechEl'])):
    print(l)
    el = radarStruct['txMechEl'][l]
    zq = np.sqrt(r**2+(4/3*a)**2+8/3*r*a*np.sin(el*np.pi/180))-4/3*a
    zqs[l,:,:] = zq
    mcp = 4
    gPwr = np.zeros((nRx,len(xq),len(yq)))
    gVr = np.zeros((nRx,len(xq),len(yq)))
    for i in range(nRx):
        ba = getBa(xx[i][:,l,:],yy[i][:,l,:],zz[i][:,l,:],rxPos2[:,i])
        msk = np.any(np.stack((xx[i][:,l,:] > (roiLims[0]-margin),xx[i][:,l,:] < (roiLims[1]+margin),yy[i][:,l,:] > (roiLims[2]-margin),yy[i][:,l,:] < (roiLims[3]+margin),ba < betaThresh[0],ba > betaThresh[1],~np.isnan(vpp[i][:,l,:]))),axis=0)
        gPwr[i,:,:] = 10*np.log10(cressman(xx[i][:,l,:][msk],yy[i][:,l,:][msk],vpp[i][:,l,:][msk],xq,yq,R,mcp))
        gVr[i,:,:] = cressman(xx[i][:,l,:][msk],yy[i][:,l,:][msk],vels[i][:,l,:][msk],xq,yq,R,mcp)
                                                                     
    pwrmsk = np.all(np.vstack((np.isnan(gPwr),np.isnan(gVr))),axis=0)
    gPwr[:,pwrmsk] = np.nan
    gVr[:,pwrmsk] = np.nan
    
    npts = np.sum(~pwrmsk)
    VR = np.zeros((nRx,1,npts))
    UV = np.zeros((3,1,npts))
    MMinv = np.zeros((nRx,3,npts))
    
    
    
    ptPos = np.stack((xq[~pwrmsk],yq[~pwrmsk],zq[~pwrmsk]))
    #ptPos = np.stack((xq,yq,zq))
    gPts = ptPos[np.newaxis,:,:]-rxPos2.T[:,:,np.newaxis]
    #rilPts = ptPos - rilPos[:,np.newaxis,np.newaxis]
    #hscPts = ptPos - hscPos[:,np.newaxis,np.newaxis]
    
    az = np.pi/2 - np.arctan2(gPts[:,1,:],gPts[:,0,:])
    dist = np.sqrt(np.sum(gPts**2,axis=1))
    el = np.arcsin(gPts[:,2,:]/dist)
    beta = np.arccos(np.cos(el[:-1,:])*np.cos(el[-1,:])*np.cos(az[:-1,:]-az[-1,:]) + np.sin(el[:-1,:])*np.sin(el[-1,:]))
    
    #MMinv[0,:,:] = np.stack(((np.sin(rilAz)*np.cos(rilEl) + np.sin(ktlxAz)*np.cos(ktlxEl))/(2*np.cos(rilBeta/2)), (np.cos(rilAz)*np.cos(rilEl) + np.cos(ktlxAz)*np.cos(ktlxEl))/(2*np.cos(rilBeta/2)), (np.sin(rilEl)+np.sin(ktlxEl))/(2*np.cos(rilBeta/2))))
    for i in range(nRx-1):
        MMinv[i,:,:] = np.stack(((np.sin(az[i,:])*np.cos(el[i,:]) + np.sin(az[-1,:])*np.cos(el[-1,:]))/(2*np.cos(beta[i,:]/2)), 
                                 (np.cos(az[i,:])*np.cos(el[i,:]) + np.cos(az[-1,:])*np.cos(el[-1,:]))/(2*np.cos(beta[i,:]/2)), 
                                 (np.sin(el[i,:])+np.sin(el[-1,:]))/(2*np.cos(beta[i,:]/2))))
    MMinv[-1,:,:] = np.stack((np.sin(az[-1,:])*np.cos(el[-1,:]),np.cos(az[-1,:])*np.cos(el[-1,:]),np.sin(el[-1,:])))
    VR[:,0,:] = gVr[:,~pwrmsk]
    
    for ii in range(npts):
        try:
            MM = np.linalg.lstsq(MMinv[:,:,ii],np.identity(nRx))[0]
            UV[:,0,ii] = MM@VR[:,0,ii]
        except:
            UV[:,0,ii] = np.zeros(3)
    
    uu = np.ones(gVr[0,:,:].shape)*np.nan
    vv = np.ones(uu.shape)*np.nan
    ww = np.ones(uu.shape)*np.nan
    
    uu[~pwrmsk] = np.squeeze(UV[0,0,:])
    vv[~pwrmsk] = np.squeeze(UV[1,0,:])
    ww[~pwrmsk] = np.squeeze(UV[2,0,:])
    
    windVectors = np.zeros((3,)+ uu.shape)
    windVectors[0,:,:] = uu
    windVectors[1,:,:] = vv
    windVectors[2,:,:] = ww
    ulim = 50
    wlim = 50
    uu[abs(uu) > ulim] = np.nan
    vv[abs(vv) > ulim] = np.nan
    ww[abs(ww) > wlim] = np.nan
    us[l,:,:] = uu
    vs[l,:,:] = vv
    ws[l,:,:] = ww
# posVectors = np.stack((xq,yq,zq))
# pvrTheory = np.squeeze(np.sum(windVectors*posVectors,axis=0)/np.sqrt(np.sum(posVectors**2,axis=0)))
# xqs = np.tile(xq,(nEl,1,1))
# yqs = np.tile(yq,(nEl,1,1))
# fig = plt.figure(figsize=(16,9))
# ax = fig.add_subplot(311)
# pc = ax.pcolormesh(yqs[:,:,119]/1000,zqs[:,:,119]/1000,us[:,:,119],vmin=-40,vmax=40,cmap='pyart_balance')
# cb = plt.colorbar(pc,ax=ax)
# cb.set_label('Retrieved U Wind (m/s)')
# ax.set_aspect('equal')
# ax.set_xlim(0,30)
# ax.set_ylim(0,12)
# ax.set_xlabel('Range (km)')
# ax.set_ylabel('Altitude (km)')
# ax.set_title('Tx Simulated '+str(radarStruct['txAz'][45])+ ' Deg RHI')
# ax = fig.add_subplot(312)
# pc = ax.pcolormesh(yqs[:,:,119]/1000,zqs[:,:,119]/1000,vs[:,:,119],vmin=-40,vmax=40,cmap='pyart_balance')
# cb = plt.colorbar(pc,ax=ax)
# cb.set_label('Retrieved V Wind (m/s)')
# ax.set_aspect('equal')
# ax.set_xlim(0,30)
# ax.set_ylim(0,12)
# ax.set_xlabel('Range (km)')
# ax.set_ylabel('Altitude (km)')
# ax.set_title('Tx Simulated '+str(radarStruct['txAz'][45])+ ' Deg RHI')
# ax = fig.add_subplot(313)
# pc = ax.pcolormesh(yqs[:,:,119]/1000,zqs[:,:,119]/1000,ws[:,:,119],vmin=-40,vmax=40,cmap='pyart_balance')
# cb = plt.colorbar(pc,ax=ax)
# cb.set_label('Retrieved W Wind (m/s)')
# ax.set_aspect('equal')
# ax.set_xlim(0,30)
# ax.set_ylim(0,12)
# ax.set_xlabel('Range (km)')
# ax.set_ylabel('Altitude (km)')
# ax.set_title('Tx Simulated '+str(radarStruct['txAz'][45])+ ' Deg RHI')

#os.chdir('/Users/semmerson/Documents/python/data/may7/may_7_250m')
os.chdir('/Users/semmerson/Documents/cm1r19.10/grinder')
#fh = Dataset('line.nc', 'r')
fh = Dataset('may20.nc', 'r')
XX = fh.variables['x'][:]
YY = fh.variables['y'][:]
ZZ = fh.variables['z'][:]
#XX,YY,ZZ = np.meshgrid(XX,YY,ZZ)
UU = fh.variables['u'][:]
VV = fh.variables['v'][:]
WW = fh.variables['w'][:]
rref = fh.variables['reflectivity'][:]
fh.close()
alt = 1
# msk = zz[:,:,alt] < 2
# WW[msk] = np.nan
# UU[msk] = np.nan
# VV[msk] = np.nan



# ax=fig.add_subplot(223)
# pc = ax.pcolormesh(XX-wrfOffset[0]-txPos[0]/1000,YY-wrfOffset[1]-txPos[1]/1000,WW[:,:,alt].T,vmin=-10,vmax=10,cmap='pyart_balance',shading='auto')
# cb = plt.colorbar(pc,ax=ax)
# cb.set_label('Model Vertical Velocity (m/s)')
# #ax.quiver(yq[::2,::2]/1000,xq[::2,::2]/1000,vv[::2,::2],uu[::2,::2],scale=1000,width=0.0015)
# #ax.quiver(XX[::2]-wrfOffset[0]-txPos[0]/1000,YY[::2]-wrfOffset[1]-txPos[1]/1000,UU[::2,::2,0].T,VV[::2,::2,0].T,scale=1000,width=0.0015)
# ax.set_xlim(plims)
# ax.set_ylim(plims)
# ax.scatter(x=0,y=0,marker='x',color='k',s=100)
# ax.scatter(x=rxPos2[1,0]/1000,y=rxPos2[0,0]/1000,marker='+',color='r',s=100)
# ax.scatter(x=rxPos2[1,1]/1000,y=rxPos2[0,1]/1000,marker='+',color='b',s=100)
# ax=fig.add_subplot(224)
# pc = ax.pcolormesh(yq/1000,xq/1000,windVectors[2,:,:]/8,vmin=-10,vmax=10,cmap='pyart_balance',shading='auto')
# cb = plt.colorbar(pc,ax=ax)
# cb.set_label('Derived Vertical Velocity (m/s)')
# #ax.quiver(yq[::s,::s]/1000,xq[::s,::s]/1000,uu[::s,::s],vv[::s,::s],scale=1000,width=0.0015)
# #plt.quiver(XX[::2]-wrfOffset[0]-txPos[0]/1000,YY[::2]-wrfOffset[1]-txPos[1]/1000,UU[::2,::2,0].T,VV[::2,::2,0].T,color='blue',scale=1000,width=0.0015)
# ax.set_xlim(plims)
# ax.set_ylim(plims)
# ax.scatter(x=0,y=0,marker='x',color='k',s=100)
# ax.scatter(x=rxPos2[1,0]/1000,y=rxPos2[0,0]/1000,marker='+',color='r',s=100)
# ax.scatter(x=rxPos2[1,1]/1000,y=rxPos2[0,1]/1000,marker='+',color='b',s=100)
# plt.tight_layout()
# kv = np.copy(ktlxVr)
# kv[np.isnan(pvrTheory)]=np.nan
alts = zq+150
alts[alts >= np.max(ZZ)*1e3] = np.max(ZZ)*1e3-1
#alts[alts <= np.max(ZZ)*1e3] = np.min(ZZ)*1e3+1
rgi = interpolate.RegularGridInterpolator((XX-wrfOffset[0]-txPos[0]/1000,YY-wrfOffset[1]-txPos[1]/1000,ZZ-txPos[2]/1000),UU)
uinterp = np.reshape(rgi(np.array([xq.flatten()/1000,yq.flatten()/1000,alts.flatten()/1000]).T),xq.shape)
rgi = interpolate.RegularGridInterpolator((XX-wrfOffset[0]-txPos[0]/1000,YY-wrfOffset[1]-txPos[1]/1000,ZZ-txPos[2]/1000),VV)
vinterp = np.reshape(rgi(np.array([xq.flatten()/1000,yq.flatten()/1000,alts.flatten()/1000]).T),xq.shape)
rgi = interpolate.RegularGridInterpolator((XX-wrfOffset[0]-txPos[0]/1000,YY-wrfOffset[1]-txPos[1]/1000,ZZ-txPos[2]/1000),WW)
winterp = np.reshape(rgi(np.array([xq.flatten()/1000,yq.flatten()/1000,alts.flatten()/1000]).T),xq.shape)
rgi = interpolate.RegularGridInterpolator((XX-wrfOffset[0]-txPos[0]/1000,YY-wrfOffset[1]-txPos[1]/1000,ZZ-txPos[2]/1000),rref)
zinterp = np.reshape(rgi(np.array([xq.flatten()/1000,yq.flatten()/1000,alts.flatten()/1000]).T),xq.shape)


xc = 0
yc = 14
window = 15
x = (xc-window, xc+window)
y = (yc-window, yc+window)
s = 4
fig = plt.figure(figsize=(14,6))
ax=fig.add_subplot(121)
pc = ax.pcolormesh(xq/1000,yq/1000,zinterp,vmin=0,vmax=60,cmap='pyart_HomeyerRainbow',shading='auto')
#pc = ax.pcolormesh(xq/1000,yq/1000,winterp,vmin=-40,vmax=40,cmap='pyart_balance',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('Model Reflectivity (dBZ)')
#ax.quiver(yq[::2,::2]/1000,xq[::2,::2]/1000,vv[::2,::2],uu[::2,::2],scale=1000,width=0.0015)
ax.quiver(xq[::s,::s]/1000,yq[::s,::s]/1000,uinterp[::s,::s],vinterp[::s,::s],scale=1000,width=0.0015)
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Model Input Reflectivity and Horizontal Winds')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
#s = 4
ax=fig.add_subplot(122)
pc = ax.pcolormesh(xq/1000,yq/1000,np.mean(gPwr,axis=0)+145,vmin=0,vmax=60,cmap='pyart_HomeyerRainbow',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('Range-Corrected Power (dB)')
ax.quiver(xq[::s,::s]/1000,yq[::s,::s]/1000,uu[::s,::s],vv[::s,::s],scale=1000,width=0.0015)
#plt.quiver(XX[::2]-wrfOffset[0]-txPos[0]/1000,YY[::2]-wrfOffset[1]-txPos[1]/1000,UU[::2,::2,0].T,VV[::2,::2,0].T,color='blue',scale=1000,width=0.0015)
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Retrieved Range-Corrected Power and Horizontal Winds')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
plt.tight_layout()

# xc = -15
# yc = -15
# window = 15
# x = (xc-window, xc+window)
# y = (yc-window, yc+window)
fig = plt.figure(figsize=(14,12))
ax=fig.add_subplot(331)
pc = ax.pcolormesh(xq/1000,yq/1000,uinterp,vmin=-50,vmax=50,cmap='pyart_balance',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('V Wind (m/s)')
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Model Input U Wind')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
ax=fig.add_subplot(332)
pc = ax.pcolormesh(xq/1000,yq/1000,uu,vmin=-50,vmax=50,cmap='pyart_balance',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('V Wind (m/s)')
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Retrieved U Wind')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
plt.tight_layout()
ax=fig.add_subplot(333)
pc = ax.pcolormesh(xq/1000,yq/1000,uu-uinterp,vmin=-10,vmax=10,cmap='pyart_balance',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('U Wind Error (m/s)')
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Retrieved U Wind Error')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
ax=fig.add_subplot(334)
pc = ax.pcolormesh(xq/1000,yq/1000,vinterp,vmin=-50,vmax=50,cmap='pyart_balance',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('V Wind (m/s)')
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Model Input V Wind')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
ax=fig.add_subplot(335)
pc = ax.pcolormesh(xq/1000,yq/1000,vv,vmin=-50,vmax=50,cmap='pyart_balance',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('V Wind (m/s)')
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Retrieved V Wind')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
plt.tight_layout()
ax=fig.add_subplot(336)
pc = ax.pcolormesh(xq/1000,yq/1000,vv-vinterp,vmin=-10,vmax=10,cmap='pyart_balance',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('V Wind Error (m/s)')
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Retrieved V Wind Error')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
ax=fig.add_subplot(337)
pc = ax.pcolormesh(xq/1000,yq/1000,winterp,vmin=-50,vmax=50,cmap='pyart_balance',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('W Wind (m/s)')
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Model Input W Wind')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
ax=fig.add_subplot(338)
pc = ax.pcolormesh(xq/1000,yq/1000,ww,vmin=-50,vmax=50,cmap='pyart_balance',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('V Wind (m/s)')
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Retrieved W Wind')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
plt.tight_layout()
ax=fig.add_subplot(339)
pc = ax.pcolormesh(xq/1000,yq/1000,ww-winterp,vmin=-10,vmax=10,cmap='pyart_balance',shading='auto')
cb = plt.colorbar(pc,ax=ax)
cb.set_label('V Wind Error (m/s)')
ax.set_xlim(x)
ax.set_ylim(y)
ax.scatter(x=0,y=0,marker='x',color='k',s=100)
ax.scatter(x=rxPos2[0,:-1]/1000,y=rxPos2[1,:-1]/1000,marker='+',color='r',s=100)
ax.set_title('Retrieved W Wind Error')
ax.set_xlabel('Zonal Distance from Tx Radar (km)')
ax.set_ylabel('Meridional Distance from Tx Radar (km)')
plt.tight_layout()

if saveMode:
    radars = {}
    radars['vpp'] = vpp
    radars['snr'] = snr
    radars['df'] = df
    radars['vel'] = vels
    radars['xx'] = xx
    radars['yy'] = yy
    radars['zz'] = zz
    radars['br'] = br
    radars['beta'] = beta
    os.chdir('/Users/semmerson/Documents/python/data/PRS')
    if len(radarStruct['txMechEl']) > 1:
        fName = str(radarStruct['rxPos'].shape[0])+'rx_'+str(len(radarStruct['txAz']))+'az_'+str(radarStruct['txMechEl'][0])+'-'+str(radarStruct['txMechEl'][-1])+'el_'+str(radarStruct['M'])+'M'+'.xz'
    else:
        fName = str(radarStruct['rxPos'].shape[0])+'rx_'+str(len(radarStruct['txAz']))+'az_'+str(radarStruct['txMechEl'])+'el_'+str(radarStruct['M'])+'M'+'.xz'
    with lzma.open(fName,'wb') as f:
        pickle.dump(radarStruct,f)
        pickle.dump(radars,f)
    del radars
    

# def animate(t):
#     s = 2
#     plt.clf()
#     ax = fig.add_subplot(111)
#     pc = ax.pcolormesh(yqs[:,:,t]/1000,zqs[:,:,t]/1000,us[:,:,t],vmin=-50,vmax=50,cmap='pyart_balance')
#     ax.quiver(yqs[::s,::s,t]/1000,zqs[::s,::s,t]/1000,vs[::s,::s,t],ws[::s,::s,t],scale=1600,width=0.001)
#     cb = plt.colorbar(pc,ax=ax)
#     cb.set_label('Retrieved U Wind (m/s)')
#     ax.set_aspect('equal')
#     ax.set_xlim(0,30)
#     ax.set_ylim(0,12)
#     ax.set_xlabel('N-S Distance from Tx (km)')
#     ax.set_ylabel('Altitude (km)')
#     #ax.set_title('Tx Simulated '+str(radarStruct['txAz'][t])+ ' Deg RHI')
#     plt.tight_layout()

    
# #os.chdir('/Users/semmerson/Documents/python/data')
# fig = plt.figure(figsize=(12,4))
# anim = mani.FuncAnimation(fig, animate, interval=1, frames=241)
# anim.save('retrieval.gif',writer='imagemagick',fps=24)

# fig = plt.figure(figsize=(12,4))
# t = 119
# s = 1
# ax = fig.add_subplot(111)
# pc = ax.pcolormesh(yqs[:,:,t]/1000,zqs[:,:,t]/1000,us[:,:,t],vmin=-50,vmax=50,cmap='pyart_balance')
# ax.quiver(yqs[::s,::s,t]/1000,zqs[::s,::s,t]/1000,vs[::s,::s,t],ws[::s,::s,t],scale=1600,width=0.001)
# cb = plt.colorbar(pc,ax=ax)
# cb.set_label('Retrieved U Wind (m/s)')
# ax.set_aspect('equal')
# ax.set_xlim(0,10)
# ax.set_ylim(0,5)
# ax.set_xlabel('N-S Distance from Tx (km)')
# ax.set_ylabel('Altitude (km)')
# #ax.set_title('Tx Simulated '+str(radarStruct['txAz'][t])+ ' Deg RHI')
# plt.tight_layout()
    
    