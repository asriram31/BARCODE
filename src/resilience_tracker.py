from reader import read_file
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import imageio.v3 as iio
from nd2reader import ND2Reader
import math, pims, yaml, gc, csv, os, glob, pickle, functools, builtins
from numpy.polynomial import Polynomial

from skimage import measure, io
from skimage.measure import label, regionprops

from scipy import ndimage

def check_span(frame):

    def check_connected(frame, axis=0):
        # Ensures that either connected across left-right or up-down axis
        if axis == 0:
            first = (frame[0] == 1).any()
            last = (frame[-1] == 1).any()
        elif axis == 1:
            first = (frame[:,0] == 1).any()
            last = (frame[:,-1] == 1).any()
        else:
            raise Exception("Axis must be 0 or 1.")
    
        struct = ndimage.generate_binary_structure(2, 2)
    
        frame_connections, num_features = ndimage.label(input=frame, structure=struct)
    
        if axis == 0:
            labeled_first = np.unique(frame_connections[0,:])
            labeled_last = np.unique(frame_connections[-1,:])
    
        if axis == 1:
            labeled_first = np.unique(frame_connections[:,0])
            labeled_last = np.unique(frame_connections[:,-1])
    
        labeled_first = set(labeled_first[labeled_first != 0])
        labeled_last = set(labeled_last[labeled_last != 0])
    
        if labeled_first.intersection(labeled_last):
            return 1
        else:
            return 0
    
    return (check_connected(frame, axis = 0) or check_connected(frame, axis = 1))

def track_void(image, name, threshold, step, save_intermediates):
    downsample = 4
    xindices = np.arange(0, image[0].shape[0], downsample)
    yindices = np.arange(0, image[0].shape[1], downsample)
    def binarize(frame, offset_threshold):
        avg_intensity = np.mean(frame)
        threshold = avg_intensity * (1 + offset_threshold)
        new_frame = np.where(frame < threshold, 0, 1)
        return new_frame
        
    def find_largest_void(frame, find_void = True):      
        if find_void:
            frame = np.invert(frame)
        labeled, a = label(frame, connectivity= 2, return_num =True) # identify the regions of connectivity 2
        regions = regionprops(labeled) # determines the region properties of the labeled
        largest_region = max(regions, key = lambda r: r.area) # determines the region with the maximum area
        return largest_region.area # returns largest region area

    def largest_island_position(frame):      
        labeled, a = label(frame, connectivity = 2, return_num =True) # identify the regions of connectivity 2
        regions = regionprops(labeled) # determines the region properties of the labeled
        largest_region = max(regions, key = lambda r: r.area) # determines the region with the maximum area
        return largest_region.centroid # returns largest region area
    
    def find_largest_void_regions(frame):
        return max(find_largest_void_mid(frame, find_void = True), find_largest_void_mid(frame, find_void = False))
        
    if save_intermediates:
        filename = os.path.join(name, 'BinarizationData.csv')
        f = open(filename, 'w')
        csvwriter = csv.writer(f)
        
    void_lst = []
    island_area_lst = []
    island_position_lst = []
    connected_lst = []
    
    for i in range(0, len(image), step):
        new_image = image[i][xindices][:,yindices]
        new_frame = binarize(new_image, threshold)
        if save_intermediates:
            csvwriter.writerow([str(i)])
            csvwriter.writerows(new_frame)
            csvwriter.writerow([])
        
        void_lst.append(find_largest_void(new_frame))
        island_area_lst.append(find_largest_void(new_frame, find_void = False))
        island_position_lst.append(largest_island_position(new_frame))
        connected_lst.append(check_span(new_frame))
    i = len(image) - 1    
    if i % step != 0:
        new_image = image[i][xindices][:,yindices]
        new_frame = binarize(new_image, threshold)
        if save_intermediates:
            csvwriter.writerow([str(i)])
            csvwriter.writerows(new_frame)
            csvwriter.writerow([])
        
        void_lst.append(find_largest_void(new_frame))
        island_area_lst.append(find_largest_void(new_frame, find_void = False))
        island_position_lst.append(largest_island_position(new_frame))
        connected_lst.append(check_span(new_frame))

        

    if save_intermediates:
        f.close()

    return void_lst, island_area_lst, island_position_lst, connected_lst

def check_resilience(file, name, channel, R_offset, frame_step, frame_start_percent, frame_stop_percent, save_intermediates, verbose):
    print = functools.partial(builtins.print, flush=True)
    vprint = print if verbose else lambda *a, **k: None
    vprint('Beginning Resilience Testing...')
    #Note for parameters: frame_step (stepsize) used to reduce the runtime. 
    image = file[:,:,:,channel]
    frame_initial_percent = 0.05

    fig, ax = plt.subplots(figsize = (5,5))

    # Error Checking: Empty Image
    if (image == 0).all():
        return [None] * 6
    
    while len(image) <= frame_step:
        frame_step = frame_step / 5
    
    largest_void_lst, island_area_lst, island_position_lst, connected_lst = track_void(image, name, R_offset, frame_step, save_intermediates)
    start_index = int(np.floor(len(image) * frame_start_percent / frame_step))
    stop_index = int(np.ceil(len(largest_void_lst) * frame_stop_percent))
    start_initial_index = int(np.ceil(len(image)*frame_initial_percent / frame_step))

    percent_gain_initial_list = np.mean(largest_void_lst[0:start_initial_index])
    percent_gain_list = np.array(largest_void_lst)/percent_gain_initial_list
    
    ax.plot(np.arange(start_index, stop_index), percent_gain_list[start_index:stop_index])
    ticks_adj = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * frame_step))
    ax.xaxis.set_major_formatter(ticks_adj)
    ax.set_xticks(np.arange(start_index, stop_index))

    ax.set_xlabel("Frames")
    ax.set_ylabel("Proportion of orginal void size")
    #Calculate
    
    avg_percent_change = np.mean(largest_void_lst[start_index:stop_index])/percent_gain_initial_list
    max_void_size = max(largest_void_lst)/(len(image[0,0,:])*len(image[0,:,0]))
    island_size = max(island_area_lst)/(len(image[0,0,:])*len(image[0,:,0]))
    island_movement = np.array(island_position_lst)[:-1,:] - np.array(island_position_lst)[1:,:]
    island_speed = np.linalg.norm(island_movement,axis = 1)
    island_direction = np.arctan2(island_movement[:,1],island_movement[:,0])
    thresh_speed = 15
    while len(island_direction[np.where(island_speed < thresh_speed)]) == 0:
        thresh_speed += 1
    island_direction = island_direction[np.where(island_speed < thresh_speed)]
    average_direction = np.average(island_direction)
    
    spanning = len([con for con in connected_lst if con == 1])/len(connected_lst)
    
    return fig, max_void_size, spanning, island_size, average_direction, avg_percent_change