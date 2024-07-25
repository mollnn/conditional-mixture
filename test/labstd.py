import numpy as np


def omega2sp(x,y,z):
    r = (x**2 + y**2 + z**2) ** 0.5
    return np.fmod(np.arctan2(z/r,x/r)/2/np.pi + 1, 1),np.arccos(y/r)/np.pi

def omega2sp_l(x,y,z):
    r = (x**2 + y**2 + z**2) ** 0.5
    return np.array([np.fmod(np.arctan2(z/r,x/r)/2/np.pi + 1, 1),np.arccos(y/r)/np.pi])

def omega_l2sp_l(a):
    x=a[0]
    y=a[1]
    z=a[2]
    r = (x**2 + y**2 + z**2) ** 0.5
    return np.array([np.fmod(np.arctan2(z/r,x/r)/2/np.pi + 1, 1),np.arccos(y/r)/np.pi])

def sp2omega(u,v):
    return np.sin(v*np.pi) * np.cos(u*2*np.pi), np.cos(v*np.pi), np.sin(v*np.pi) * np.sin(u*2*np.pi)

def sp2omega_l(u,v):
    return np.array([np.sin(v*np.pi) * np.cos(u*2*np.pi), np.cos(v*np.pi), np.sin(v*np.pi) * np.sin(u*2*np.pi)])

def sp_l2omega_l(a):
    u = np.fmod(np.fmod(a[0], 1) + 1, 1)
    v = np.fmod(np.fmod(a[1], 1) + 1, 1)
    return np.array([np.sin(v*np.pi) * np.cos(u*2*np.pi), np.cos(v*np.pi), np.sin(v*np.pi) * np.sin(u*2*np.pi)])
