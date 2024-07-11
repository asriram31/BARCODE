from reader import read_file
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.fft import fft2, ifft2
from scipy.interpolate import Akima1DInterpolator
from scipy import optimize
import os, math

#Find the first point at which a spline interpolant for a set of correlator values descends below a certain threshold
def findRoot(xValues, yValues,threshold):
    interpolator = Akima1DInterpolator(xValues, yValues)
    return optimize.root_scalar(lambda arg: interpolator([arg])[0]-threshold ,bracket=[min(xValues),max(xValues)]).root


def divergence_npgrad(flow):
    flow = np.swapaxes(flow, 0, 1)
    Fx, Fy = flow[:, :, 0], flow[:, :, 1]
    dFx_dx = np.gradient(Fx, axis=0)
    dFy_dy = np.gradient(Fy, axis=1)
    return dFx_dx + dFy_dy

def check_flow(file, name, channel, min_corr_len, min_fraction, frame_stride, downsample, pix_size, bin_width, decay_threshold = 1/np.exp(1), tbf=1):
    #Width of annuli in pixels
    pixel_bin_width = np.ceil(bin_width / pix_size)
    #Length at which to stop computing correlators
    max_len = 500
    #Length in pixels at which to cut off correlators
    max_pixel_len = np.rint(max_len / pix_size)
    #Cutoff magnitude to consider a vector to be null; also helps to avoid divide-by-zero errors
    flt_tol = 1e-10
    def execute_opt_flow(images, start, stop, divs, xMeans, yMeans, vxMeans, vyMeans, corrLens, pos, xindices, yindices):
        def normalVectors(velocities):
            #Find velocity directions
            def normalize(vector):
                magnitude = np.linalg.norm(vector)
                if magnitude == 0: return np.array([0,0])
                return np.where(magnitude > flt_tol, np.array(vector)/magnitude, np.array([0, 0]))
                    
            normals = np.zeros_like(velocities)
            for i in range(0, velocities.shape[0]):
                for j in range(0, velocities.shape[1]):
                    normals[i][j] = normalize(velocities[i][j])
        
            return normals
            
        flow = cv.calcOpticalFlowFarneback(images[start], images[stop], None, 0.5, 3, 32, 3, 5, 1.2, 0)
        divs = np.append(divs, divergence_npgrad(flow))
        # curls = np.append(curls, curl_npgrad(flow))
        directions = normalVectors(flow[xindices][:,yindices])
        dirX = directions[:,:,0]
        dirY = directions[:,:,1]
        xMeans = np.append(xMeans, dirX.mean())
        yMeans = np.append(yMeans, dirY.mean())
        downU = flow[:,:,0][xindices][:,yindices]
        downU = np.flipud(downU)
        downV = -1*flow[:,:,1][xindices][:,yindices]
        downV = np.flipud(downV)
        if np.isin(beg, positions):
            fig2, ax2 = plt.subplots(figsize=(10,10))
            q = ax2.quiver(xindices, yindices, downU, downV,color='blue')
            figpath2 = os.path.join(name,  'Frame '+ str(beg) + ' Flow Field.png')
            print(figpath2)
            fig2.savefig(figpath2)
            plt.close(fig2)
        vxMeans = np.append(vxMeans, downU.mean())
        vyMeans = np.append(vyMeans, downV.mean())        
        xFFT = fft2(dirX)
        xConv = np.real(ifft2(np.multiply(xFFT,np.conjugate(xFFT))))
        yFFT = fft2(dirY)
        yConv = np.real(ifft2(np.multiply(yFFT,np.conjugate(yFFT))))
        convSum = np.add(xConv,yConv)
        means = np.array([])
        inRadii = np.array([])
        for i in range(0,int(max_pixel_len),int(pixel_bin_width)):
                inBin = convSum[(radii > i -.5*pixel_bin_width)&(radii < i + .5*pixel_bin_width)]
                if(len(inBin) > 0):
                        inRadii = np.append(inRadii, i)
                        means = np.append(means, inBin.mean()/convSum[0,0])
        try:
            corrLens[pos] = pix_size*findRoot(inRadii,means,decay_threshold)
        except ValueError:
            corrLens[pos] = 0
    
        return [xMeans, yMeans, vxMeans, vyMeans, divs]

    fig, ax = plt.subplots(figsize=(5,5))

    images = file[:,:,:,channel]

    positions = np.array([0, int(np.floor(len(images)/2)), len(images) - frame_stride])

    # Error Checking: Empty Images
    if (images == 0).all():
       verdict = "Data not available for this channel."
       return verdict, fig

    xindices = np.arange(0, images[0].shape[0], downsample)
    yindices = np.arange(0, images[0].shape[1], downsample)

    radii = np.zeros((len(xindices),len(yindices)))
    for i in range(0,len(xindices)):
        for j in range(0,len(yindices)):
            radii[i][j] = np.sqrt(xindices[i]**2 + yindices[j]**2)

    #For each consecutive pair
    corrLens = np.zeros(len(images)-frame_stride)
    pos = 0
    xMeans = np.array([])
    yMeans = np.array([])
    vxMeans = np.array([])
    vyMeans = np.array([])
    divs = np.array([])
    
    for beg in range(0,len(images)-frame_stride,frame_stride):
        end = beg + frame_stride
        arr = execute_opt_flow(images, beg, end, divs, xMeans, yMeans, vxMeans, vyMeans, corrLens, pos, xindices, yindices)
        xMeans, yMeans, vxMeans, vyMeans, divs = arr
        pos += 1

    beg = len(images) - frame_stride
    end = len(images) - 1
    arr = execute_opt_flow(images, beg, end, divs, xMeans, yMeans, vxMeans, vyMeans, corrLens, pos, xindices, yindices)
    xMeans, yMeans, vxMeans, vyMeans, divs = arr
    
    if(len(corrLens[corrLens>min_corr_len])/len(corrLens) > min_fraction):
        verdict = 1
    else:
        verdict = 0
    ax.plot(range(0,len(corrLens)),corrLens)
    direct = math.atan2(yMeans.mean(), xMeans.mean())
    mean_div = divs.mean()
    # print("x dir: ", xMeans.mean(), "\n","y direc: ", yMeans.mean(), "\n","vx mean: ", vxMeans.mean(), "\n","vy mean: ", vyMeans.mean(), "\n", "angle:", direct, "\n", "divergence mean:", mean_div)
    avg_vel = (vxMeans.mean() ** 2 + vyMeans.mean() ** 2) ** (1/2)
    
    return verdict, fig, direct, avg_vel, mean_div
