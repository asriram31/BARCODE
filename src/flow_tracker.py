from reader import read_file
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.fft import fft2, ifft2
from scipy.interpolate import Akima1DInterpolator
from scipy import optimize
import os, math, csv, functools, builtins
import matplotlib.ticker as ticker
import statistics

def divergence_npgrad(flow):
    flow = np.swapaxes(flow, 0, 1)
    Fx, Fy = flow[:, :, 0], flow[:, :, 1]
    dFx_dx = np.gradient(Fx, axis=0)
    dFy_dy = np.gradient(Fy, axis=1)
    return dFx_dx + dFy_dy

def check_flow(file, name, channel, frame_stride, downsample, frame_interval, nm_pix_ratio, return_graphs, save_intermediates, verbose):
    print = functools.partial(builtins.print, flush=True)
    vprint = print if verbose else lambda *a, **k: None
    vprint('Beginning Flow Testing')
    #Cutoff magnitude to consider a vector to be null; also helps to avoid divide-by-zero errors
    flt_tol = 1e-10
    def execute_opt_flow(images, start, stop, divs, dirMeans, dirSDs, vxMeans, vyMeans, speeds, pos, xindices, yindices, save_intermediates, writer):
        flow = cv.calcOpticalFlowFarneback(images[start], images[stop], None, 0.5, 3, 32, 3, 5, 1.2, 0)
        divs = np.append(divs, divergence_npgrad(flow))
        
        downU = flow[:,:,0][xindices][:,yindices]
        downU = np.flipud(downU)
        downV = -1*flow[:,:,1][xindices][:,yindices]
        downV = np.flipud(downV)

        directions = np.arctan2(downV, downU)
        dirMean = directions.mean()
        if save_intermediates:
            writer.writerow(["Flow Field (" + str(beg) + "-" + str(end) + ")"])
            writer.writerow(["X-Direction"])
            writer.writerows(downU)
            writer.writerow(["Y-Direction"])
            writer.writerows(downV)
        speed = (downU ** 2 + downV ** 2) ** (1/2)
        if np.isin(beg, positions) and return_graphs:
            fig, ax = plt.subplots(figsize=(10,10))
            q = ax.quiver(downU, downV, color='blue')
            figpath = os.path.join(name,  'Frame '+ str(beg) + ' Flow Field.png')
            ticks_adj = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * downsample))
            ax.xaxis.set_major_formatter(ticks_adj)
            ax.yaxis.set_major_formatter(ticks_adj)
            fig.savefig(figpath)
            plt.close(fig)

        dirMeans = np.append(dirMeans, directions.mean())
        dirSDs = np.append(dirSDs, np.std(directions))
        vxMeans = np.append(vxMeans, downU.mean())
        vyMeans = np.append(vyMeans, downV.mean())        
        speeds = np.append(speeds, speed.mean())
        return [dirMeans, dirSDs, vxMeans, vyMeans, speeds, divs]


    images = file[:,:,:,channel]

    end_point = len(images) - frame_stride
    while end_point <= 0: # Checking to see if frame_stride is too large
        frame_stride = int(np.ceil(frame_stride / 5))
        vprint('Flow field frame step too large for video, dynamically adjusting, new frame step:', frame_stride)
        end_point = len(images) - frame_stride

    positions = np.array([0, int(np.floor(len(images)/2)), end_point])

    # Error Checking: Empty Images
    if (images == 0).all():
       return [None] * 5

    xindices = np.arange(0, images[0].shape[0], downsample)
    yindices = np.arange(0, images[0].shape[1], downsample)

    radii = np.zeros((len(xindices),len(yindices)))
    for i in range(0,len(xindices)):
        for j in range(0,len(yindices)):
            radii[i][j] = np.sqrt(xindices[i]**2 + yindices[j]**2)

    #For each consecutive pair
    pos = 0
    dirMeans = np.array([])
    dirSDs = np.array([])
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
        arr = execute_opt_flow(images, beg, end, divs, dirMeans, dirSDs, vxMeans, vyMeans, speeds, pos, xindices, yindices, save_intermediates, csvwriter)
        dirMeans, dirSDs, vxMeans, vyMeans, speeds, divs = arr
        pos += 1
    if end_point != len(images) - 1:
        beg = end
        end = len(images) - 1
        arr = execute_opt_flow(images, beg, end, divs, dirMeans, dirSDs, vxMeans, vyMeans, speeds, pos, xindices, yindices, save_intermediates, csvwriter)
        dirMeans, dirSDs, vxMeans, vyMeans, speeds, divs = arr

    if save_intermediates:
        myfile.close()
    direct = dirMeans.mean()
    directSD = dirSDs.mean()
    mean_div = divs.mean()
    mean_vel = (vxMeans.mean() ** 2 + vyMeans.mean() ** 2) ** (1/2)
    mean_speed = speeds.mean()
    
    # Corrections to convert from pixels/flow field -> nm / sec
    mean_vel = mean_vel * (1 / frame_stride) * nm_pix_ratio * (1 / frame_interval)
    mean_speed = mean_speed * (1 / frame_stride) * nm_pix_ratio * (1 / frame_interval)
    
    return direct, directSD, mean_vel, mean_speed, mean_div
