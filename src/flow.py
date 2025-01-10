import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os, csv, functools, builtins
import matplotlib.ticker as ticker

"""
Takes an average downsampling of 2D array to go from array of dimension (x, y) to (x/N, y/N)
@param arr: the input array of shape (x, y)
@param N: the number to downsample the array

@return: an output array of shape (x/N, y/N)
"""
def groupAvg(arr, N):
    result = np.cumsum(arr, 0)[N-1::N]/float(N) # Average all rows in array
    result = np.cumsum(result, 1)[:,N-1::N]/float(N) # Average all columns in array
    result[1:] = result[1:] - result[:-1]
    result[:,1:] = result[:,1:] - result[:,:-1]
    return result

"""
Takes an input video file and uses flow fields between frames to extract metrics
@param file: the input video file
@param name: the name of the output directory folder for video intermediates + plots
@param channel: the channel to analyze the video
@param frame_stride: the interval (in frames) between flow fields
@param downsample: controls the spatial downsampling rate of the flow fields; used to control noise
@param frame_interval: the interval (in units of time) between frames (seconds/frame); used to convert to real units
@param nm_pix_ratio: the ratio between nm to pixels; used to convert to real units
@param return_graphs: controls whether sample flow fields are saved to output directory as images
@param save_intermediates: controls whether all flow fields are saved as CSV file for further analysis
@param verbose: controls the level of printed output to the GUI
@param winsize: controls the window size for Farneback polynomial approximation

@return: a list containing the average speed, change in speed, direction, and directional spread
"""
def check_flow(file, name, channel, frame_stride, downsample, frame_interval, nm_pix_ratio, return_graphs, save_intermediates, verbose, winsize = 16):
    # Defines print to enable printing only if verbose setting set to True
    print = functools.partial(builtins.print, flush=True)
    vprint = print if verbose else lambda *a, **k: None
    vprint('Beginning Flow Testing')

    """
    Calculates average speed, average direction, directional spread for each flow field
    @param images
    @param start
    @param stop
    @param pos
    @param theta
    @param sigma_theta
    @param speeds
    @param writer
    """
    def execute_opt_flow(images, start, stop, pos, thetas, sigma_thetas, speeds, writer):
        flow = cv.calcOpticalFlowFarneback(images[start], images[stop], None, 0.5, 3, winsize, 3, 5, 1.2, 0)
        flow_reduced = groupAvg(flow, downsample)
        downU = flow_reduced[:,:,0]
        downV = flow_reduced[:,:,1]
        downU = np.flipud(downU)
        downV = np.flipud(downV)

        directions = np.arctan2(downV, downU)
        if save_intermediates:
            writer.writerow(["Flow Field (" + str(beg) + "-" + str(end) + ")"])
            writer.writerow(["X-Direction"])
            writer.writerows(downU)
            writer.writerow(["Y-Direction"])
            writer.writerows(downV)
        speed = (downU ** 2 + downV ** 2) ** (1/2)

        mid_point_arr = range(0, end_point, frame_stride)
        mid_point = mid_point_arr[int((len(mid_point_arr) - 1)/2)]
        positions = np.array([0, mid_point, end_point])
        img_shape = downU.shape[0] / downU.shape[1]
        if np.isin(beg, positions) and return_graphs:
            fig, ax = plt.subplots(figsize=(10 * img_shape,10))
            q = ax.quiver(downU, downV, color='blue')
            figpath = os.path.join(name,  'Frame '+ str(beg) + ' Flow Field.png')
            ticks_adj = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * downsample))
            ax.xaxis.set_major_formatter(ticks_adj)
            ax.yaxis.set_major_formatter(ticks_adj)
            fig.savefig(figpath)
            plt.close(fig)
        
        # Convert speed from pixels / interval to nm/sec
        # Conversion: px/interval * interval/frame * 1/(sec/frame) * nm/px
        avg_speed = np.mean(speed) * 1/(frame_interval) * 1/(frame_stride) * nm_pix_ratio
        thetas.append(np.mean(directions))
        sigma_thetas.append(np.std(directions))
        speeds.append(avg_speed)
        return

    images = file[:,:,:,channel]
    # Error Checking: Empty Images
    if (images == 0).all():
       return [np.nan] * 4

    end_point = len(images) - frame_stride
    while end_point <= 0: # Checking to see if frame_stride is too large
        frame_stride = int(np.ceil(frame_stride / 5))
        vprint('Flow field frame step too large for video, dynamically adjusting, new frame step:', frame_stride)
        end_point = len(images) - frame_stride


    thetas = []
    sigma_thetas = []
    speeds = []
    pos = 0
    filename = os.path.join(name, 'OpticalFlow.csv')

    # Prepares the intermediate file for saving if setting is turned on
    if save_intermediates:
        myfile = open(filename, "w")
        csvwriter = csv.writer(myfile)
    else: 
        csvwriter = None
    
    #For each consecutive pairs of frames, calculate the metrics average speed, average direction, directional spread
    for beg in range(0, end_point, frame_stride):
        end = beg + frame_stride
        execute_opt_flow(images, beg, end, pos, thetas, sigma_thetas, speeds, csvwriter)
        pos += 1
    # If interval between frames does not reach end of video, add additional calculation step
    if end_point != len(images) - 1:
        beg = end
        end = len(images) - 1
        execute_opt_flow(images, beg, end, pos, thetas, sigma_thetas, speeds, csvwriter)
        
    # Close the CSV intermediate file
    if save_intermediates:
        myfile.close()


    thetas = np.array(thetas)
    sigma_thetas = np.array(sigma_thetas)
    speeds = np.array(speeds)    
    theta = thetas.mean() # Metric for average direction of flow (-pi, pi) # "Flow Direction"
    sigma_theta = sigma_thetas.mean() # Metric for st. dev of flow (-pi, pi) # "Flow Directional Spread"
    mean_speed = speeds.mean() # Metric for avg. speed (units of nm/s) # Average speed
    # Calculate delta speed as (v_f - v_i) / (t_f - t_i)
    delta_speed = speeds[-1] - speeds[0]
    return [mean_speed, delta_speed, theta, sigma_theta]
