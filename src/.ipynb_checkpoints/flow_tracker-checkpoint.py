from reader import read_file
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.fft import fft2, ifft2
from scipy.interpolate import Akima1DInterpolator
from scipy import optimize
import os, math, csv, functools, builtins

def divergence_npgrad(flow):
    flow = np.swapaxes(flow, 0, 1)
    Fx, Fy = flow[:, :, 0], flow[:, :, 1]
    dFx_dx = np.gradient(Fx, axis=0)
    dFy_dy = np.gradient(Fy, axis=1)
    return dFx_dx + dFy_dy

def check_flow(file, name, channel, frame_stride, downsample, return_graphs, save_intermediates, verbose):
    print = functools.partial(builtins.print, flush=True)
    vprint = print if verbose else lambda *a, **k: None
    vprint('Beginning Flow Testing')
    #Cutoff magnitude to consider a vector to be null; also helps to avoid divide-by-zero errors
    flt_tol = 1e-10
    def execute_opt_flow(images, start, stop, divs, xMeans, yMeans, vxMeans, vyMeans, speeds, pos, xindices, yindices, save_intermediates, writer):
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
        if save_intermediates:
            writer.writerow(["Flow Field (" + str(beg) + "-" + str(end) + ")"])
            writer.writerow(["X-Direction"])
            writer.writerows(downU)
            writer.writerow(["Y-Direction"])
            writer.writerows(downV)
        speed = (downU ** 2 + downV ** 2) ** (1/2)
        if np.isin(beg, positions) and return_graphs:
            fig2, ax2 = plt.subplots(figsize=(10,10))
            q = ax2.quiver(xindices, yindices, downU, downV,color='blue')
            figpath2 = os.path.join(name,  'Frame '+ str(beg) + ' Flow Field.png')
            fig2.savefig(figpath2)
            plt.close(fig2)
        vxMeans = np.append(vxMeans, downU.mean())
        vyMeans = np.append(vyMeans, downV.mean())        
        speeds = np.append(speeds, speed.mean())
        return [xMeans, yMeans, vxMeans, vyMeans, speeds, divs]


    images = file[:,:,:,channel]

    end_point = len(images) - frame_stride
    while end_point <= 0: # Checking to see if frame_stride is too large
        frame_stride = int(np.ceil(frame_stride / 5))
        vprint('Flow field frame step too large for video, dynamically adjusting, new frame step:', frame_stride)
        end_point = len(images) - frame_stride

    positions = np.array([0, int(np.floor(len(images)/2)), end_point])

    # Error Checking: Empty Images
    if (images == 0).all():
       return [None] * 4

    xindices = np.arange(0, images[0].shape[0], downsample)
    yindices = np.arange(0, images[0].shape[1], downsample)

    radii = np.zeros((len(xindices),len(yindices)))
    for i in range(0,len(xindices)):
        for j in range(0,len(yindices)):
            radii[i][j] = np.sqrt(xindices[i]**2 + yindices[j]**2)

    #For each consecutive pair
    pos = 0
    xMeans = np.array([])
    yMeans = np.array([])
    vxMeans = np.array([])
    vyMeans = np.array([])
    speeds = np.array([])
    divs = np.array([])

    filename = os.path.join(name, 'OpticalFlow.csv')
    if save_intermediates:
        myfile = open(filename, "w")
        csvwriter = csv.writer(myfile)

    else: csvwriter = None
    
    for beg in range(0, end_point, frame_stride):
        end = beg + frame_stride
        arr = execute_opt_flow(images, beg, end, divs, xMeans, yMeans, vxMeans, vyMeans, speeds, pos, xindices, yindices, save_intermediates, csvwriter)
        xMeans, yMeans, vxMeans, vyMeans, speeds, divs = arr
        pos += 1
    if end_point != len(images) - 1:
        beg = end
        end = len(images) - 1
        arr = execute_opt_flow(images, beg, end, divs, xMeans, yMeans, vxMeans, vyMeans, speeds, pos, xindices, yindices, save_intermediates, csvwriter)
        xMeans, yMeans, vxMeans, vyMeans, speeds, divs = arr

    if save_intermediates:
        myfile.close()
    
    direct = math.atan2(yMeans.mean(), xMeans.mean())
    mean_div = divs.mean()
    vprint("x dir: ", xMeans.mean(), "\n","y direc: ", yMeans.mean(), "\n","vx mean: ", vxMeans.mean(), "\n","vy mean: ", vyMeans.mean(), "\n", "angle:", direct, "\n", "divergence mean:", mean_div)
    mean_vel = (vxMeans.mean() ** 2 + vyMeans.mean() ** 2) ** (1/2)
    mean_speed = speeds.mean()
    
    return direct, mean_vel, mean_speed, mean_div
