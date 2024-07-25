from matplotlib import pyplot as plt
import config
import numpy as np
import random 
import time
import copy
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import sys
from labstd import *
from tqdm import tqdm

def my_omega2sp(x,y,z):
    r = (x**2 + y**2 + z**2) ** 0.5
    
    return x/r * 0.5 + 0.5, z/r*0.5 + 0.5

def my_sp2omega(u,v):
    
    x = u * 2 - 1
    z = v * 2 - 1
    r0 = x * x + z * z
    if r0 > 1: return 0, -1, 0
    y = (1 - r0) ** 0.5
    return x, y, z

def myplot_test(ax):
    ax.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0])



def random_dir():
    x = random.normalvariate(0,1)
    y = random.normalvariate(0,1)
    z = random.normalvariate(0,1)
    r = (x**2+y**2+z**2)**0.5
    return (x/r,y/r,z/r)

def random_offset():
    x = random.normalvariate(0,1)
    y = random.normalvariate(0,1)
    z = random.normalvariate(0,1)
    r = 1
    return (x/r,y/r,z/r)
class DistributionReconstructor:
    def load(self, a, cond = None):
        
        
        return

    def sample(self):
        
        
        ans = np.array(random_dir())
        if ans[1] < 0:
            ans = -ans
        return ans

    def samples(self, n):
        ans = [self.sample() for i in range(n)]
        return np.array(ans)

    def export_map(self, resolution, n_samples):
        ans = np.zeros([resolution, resolution])
        smps = self.samples(n = n_samples)
        for i in range(n_samples):
            wx, wy, wz = smps[i]
            x, y = my_omega2sp(wx, wy, wz)
            nx, ny = int(x*resolution), int(y*resolution)
            nx, ny = 0 if nx < 0 else nx, 0 if ny < 0 else ny
            nx, ny = resolution - 1 if nx >= resolution else nx, resolution - 1 if ny >= resolution else ny
            ans[ny][nx] += 1 / n_samples
        return ans


class PhotonDR(DistributionReconstructor):
    def __init__(self):
        self.rad = []
        self.dir = []
        self.pos = []
        self.posw = []
        self.flag = 0

    def load(self, a, cond = None):
        self.rad = [i[5] for i in a]
        self.dir = [i[0:3] for i in a]
        if len(a)>0 and len(a[0])>5:
            self.pos = [i[6:9] for i in a]
        return

    def sample(self):
        
        ans = random.choices(self.dir, self.rad, k=1)[0]
        return np.array(ans)

class UniformPhotonBlurDR(PhotonDR):
    def sample(self):
        
        ans = random.choices(self.dir, self.rad, k=1)[0]
        return np.array(ans) + np.array(random_dir()) * 0.01

class AdaptivePhotonBlurDR(PhotonDR):
    def sample(self):
        
        ans = random.choices(self.dir, self.rad, k=1)[0]
        deltas = [np.array(i) - np.array(ans) for i in self.dir]
        dist2s_omega = [np.dot(i, i) for i in deltas]
        dist2s_omega = sorted(dist2s_omega)
        radius = (dist2s_omega[min(len(dist2s_omega)-1, 2)]) ** 0.5
        
        return np.array(ans) + np.array(random_dir()) * random.normalvariate(mu=0, sigma=1) * radius 

        
def myplot_photon(ax, x, y, s=100, lw=0.3):
    ax.scatter(x, y, s=s, linewidth=lw, marker='+',
               color=np.array([0,1,0]))
            

def myplot_photon_recon(ax, a, s=100, lw=0.5, MAP_NUM_SAMPLES = 10000, RES_DISTB = 200, LIMX=(0,1), LIMY=(0,1)):
    rec = AdaptivePhotonBlurDR()
    rec.load(a)
    map_guess = rec.export_map(RES_DISTB, MAP_NUM_SAMPLES)
    uvg = np.linspace(0, 1, RES_DISTB)
    map_guess_color = np.ones(( RES_DISTB, RES_DISTB))*0.0


    gcmap = copy.deepcopy(plt.cm.get_cmap('inferno', 256))
    __idx = 0
    for i in gcmap.colors:
        
        
        
        i[-1] = 0.0 + 1.0 * (__idx / 256) ** 0.5
        __idx += 1


    for j, u in enumerate(tqdm(uvg)):
        if (u < LIMX[0] or u > LIMX[1]): continue
        for i, v in enumerate(uvg):
            if (v < LIMY[0] or v > LIMY[1] or j + 1 == len(uvg) or i + 1 == len(uvg) or (u - 0.5) ** 2 + (v - 0.5) ** 2 > 0.248): 
                map_guess_color[i, j] = -10
                continue 
            map_guess_color[i, j] = np.log(np.exp(-10) + map_guess[i, j] * RES_DISTB * RES_DISTB) 
    ax0 = ax.imshow(map_guess_color, extent=[0, 0.9999, 0.9999, 0], cmap=gcmap, interpolation='bicubic')
    return map_guess


def myplot_photon_with_value(ax, dir, val, s=100, lw=0.5):
    myplot_photon(ax, x, y, s, lw)
    for xx, yy, vv in zip(x, y, val):
        ax.text(xx + 0.001, yy + 0.001, " " + str(vv), color = 'g', fontsize = 'x-small')


class BasinDescriptor:
    def __init__(self, sol_omega_, sol_uv_, sol_f_, map_idx_):
        self.sol_omega = sol_omega_
        self.sol_uv = sol_uv_
        self.sol_f = sol_f_
        self.map_idx = map_idx_

    def load(self, file):
        npz = np.load(file)
        self.sol_omega = npz["arr_0"]
        self.sol_uv = npz["arr_1"]
        self.sol_f = npz["arr_2"]
        self.map_idx = npz["arr_3"]

    def save(self, file):
        np.savez(file, self.sol_omega, self.sol_uv, self.sol_f, self.map_idx)


def myplot_basin(ax, basin_desc, value_base = -2, value_range = 2, color_bar_on = True):
    solutions = basin_desc.sol_omega
    solutions_uv = basin_desc.sol_uv
    solutions_radiance = basin_desc.sol_f
    map_idx = basin_desc.map_idx

    RES_BASIN = len(map_idx)
    uv = np.linspace(0, 1, RES_BASIN)


    if len(solutions) == 0:
        colors = []
        ax.imshow([[0]])
    else:
        from matplotlib.colors import ListedColormap,LinearSegmentedColormap
        def getDiscreteCMAP(n):
            cmap = plt.cm.PuBu_r  
            cmaplist = [np.array(cmap(i)) * 0.5 for i in range(cmap.N)]
            k = 256 // n if n > 0 else 0
            return cmaplist[::k][:n] if n > 0 else []
        colors = np.array(getDiscreteCMAP(len(solutions)))[:,:3]
        map_color = np.ones((RES_BASIN, RES_BASIN)) * 1e-5
        mx = max(solutions_radiance)
        for j, u in enumerate(tqdm(uv)):
            for i, v in enumerate(uv):
                id = int(map_idx[i, j])
                if id >= 0:
                    map_color[i, j] = solutions_radiance[id] + 0.01
                if (u * 2  - 1) ** 2 + (v * 2 - 1) ** 2 >= 1:
                    map_color[i, j] = 1e9
        cmap = plt.cm.PuBu_r
        clist=[np.array(cmap(i)) for i in range(cmap.N)]
        clist[-1] = np.array([1.0, 1.0, 1.0])
        newcmp = LinearSegmentedColormap.from_list('chaos',clist)
        plt.axis("off")
        ax.set_axis_off()
        import matplotlib as mplt
        ax0 = ax.imshow(map_color, extent=[0, 1, 1, 0], 
                  cmap=newcmp, interpolation='nearest', norm=mplt.colors.LogNorm(vmin=10**value_base,vmax=10**value_range))
        for i, s in enumerate(solutions_uv):
            ax.plot(s[0], s[1], 'o', ms=4, color='k')
            ax.plot(s[0], s[1], 'o', ms=2, color='w')
            
        if color_bar_on:
            plt.colorbar(ax0, ax=ax)
    return



def __read_knn(PTCLE_FILE, x, y, z, k):
    fp = open(PTCLE_FILE)
    a = fp.readlines()
    a = [list(map(float,i.split(" "))) for i in a]
    a = np.array(a)
    a = sorted(a, key=lambda p: (p[0]-x)**2+(p[1]-y)**2+(p[2]-z)**2)
    a = a[:k]
    return a


def solve_knn(PTCLE_FILE, shading_point, knn_k = 32, shading_uv = np.array([0, 0]), correction = False):
    ansx = []
    ansy = []
    knnres = __read_knn(PTCLE_FILE, shading_point[0],shading_point[1],shading_point[2],knn_k)
    ans = []
    for i in knnres:
        position = i[0:3]
        light_position = i[3:6]
        next_position = i[6:9]
        flux = i[9]
        duv = shading_uv - i[10:12]
        dx_du = i[12:15]
        dx_dv = i[15:18]
        
        
        def normalize(x):
            import math
            len = math.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
            return x / len
        direction = normalize(next_position - shading_point)

        omega_spframe = my_omega2sp(direction[0], direction[1], direction[2])
        
        ansx.append(omega_spframe[0])
        ansy.append(omega_spframe[1])
        delta_pos = np.array(position) - np.array(shading_point)
        delta_pos = delta_pos.tolist()
        ans.append([direction[0], direction[1], direction[2]] + [omega_spframe[0], omega_spframe[1]] + [flux] + delta_pos)
    return np.array(ans)

def myplot_imgtext(ax, img, title = '', text = ''):
    ax.set_axis_off()
    ax.imshow(img)
    ax.set_title(title)
    ax.text(4, 4, text, color='k', verticalalignment='top')
    ax.text(4, 6, text, color='k', verticalalignment='top')
    ax.text(6, 4, text, color='k', verticalalignment='top')
    ax.text(6, 6, text, color='k', verticalalignment='top')
    ax.text(5, 5, text, color='w', verticalalignment='top')



def imcrop(a, crop = {}):
    x = 0 if "x" not in crop.keys() else crop["x"]
    y = 0 if "y" not in crop.keys() else crop["y"]
    w = a.shape[1] if "w" not in crop.keys() else crop["w"]
    h = a.shape[0] if "h" not in crop.keys() else crop["h"]
    return a[y:y+h, x:x+w, :]

def readexr(a, crop = {}):
    imga = cv2.imread(a, cv2.IMREAD_UNCHANGED)
    if len(imga.shape) == 2: 
        imga = cv2.merge((imga,imga,imga))
    imga = imga.astype("float")[:, :, [2, 1, 0]]
    return imcrop(imga, crop)

def avgpool2(image):
    height, width, channels = image.shape
    image_matrix = image.reshape(height // 4, 4, width // 4, 4, channels)
    smaller_image = np.mean(image_matrix, axis=(1, 3))
    larger_image = np.repeat(np.repeat(smaller_image, 4, axis=0), 4, axis=1)
    return larger_image

def rmse(imageA, imageB, cv = 1e99):
    imageA = np.minimum(imageA, cv)
    imageB = np.minimum(imageB, cv)
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    err /= 3
    return err ** 0.5

def mse(imageA, imageB, cv = 1e99):
    imageA = np.minimum(imageA, cv)
    imageB = np.minimum(imageB, cv)
    imageA = np.maximum(imageA, 0)
    imageB = np.maximum(imageB, 0)
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    err /= 3
    return err

def mse_unsuper(iA, cv=1e2):
    iA2 = np.minimum(iA, cv)
    return mse(iA2, avgpool2(iA2), cv) 

def mae(imageA, imageB):
    err = np.sum(((imageA.astype("float") - imageB.astype("float")) ** 2) ** 0.5)
    err /= float(imageA.shape[0] * imageA.shape[1])
    err /= 3
    return err

def mape(imageA, imageB):
    err = np.sum(((imageA.astype("float") - imageB.astype("float")) ** 2) ** 0.5 / (imageB.astype("float")+ 1e-2))
    err /= float(imageA.shape[0] * imageA.shape[1])
    err /= 3
    return err

def relmse(imageA, imageB):
    mat = ((imageA.astype("float") - imageB.astype("float")) ** 2) / (imageB.astype("float") ** 2 + 1e-3)
    data = mat.reshape(-1)
    max_val = np.percentile(data, 99.9)
    max_val = 1e9
    filtered_data = data[data <= max_val]
    err = np.sum(data)
    err /= float(imageA.shape[0] * imageA.shape[1])
    err /= 3
    return err

def mae_slide(imageA, imageA1, imageB):
    err = ((imageA.astype("float") - imageB.astype("float")) ** 2) ** 0.5
    err1 = ((imageA1.astype("float") - imageB.astype("float")) ** 2) ** 0.5
    h, w = len(imageA), len(imageA[0])
    ans = 0
    for x in range(0, w - 100, 50):
        for y in range(0, h - 100, 50):
            img_crop = imcrop(err, {"x": x, "y": y, "w": 100, "h": 100})
            img_crop1 = imcrop(err1, {"x": x, "y": y, "w": 100, "h": 100})
            s = np.sum(img_crop)
            s1 = np.sum(img_crop1)
            ans = max(ans, s / s1)
    return ans
    



