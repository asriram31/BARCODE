from reader import read_file
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from nd2reader import ND2Reader
import math, pims, yaml, gc, csv, os, glob, pickle

import numpy as np
from numpy.polynomial import Polynomial

from skimage import measure, io
from skimage.measure import label, regionprops

from scipy import ndimage

def check_span(image, R_thresh):
    
    def binarize(frame, offset_threshold):
        avg_intensity = np.mean(frame)
        threshold = avg_intensity * (1 + offset_threshold)
        new_frame = np.where(frame < threshold, 0, 1)
        return new_frame
    
    def check_connected(frame, axis=0):
        # Ensures that either 
        if axis == 0:
            first = (frame[0] == 1).any()
            last = (frame[len(frame) - 1] == 1).any()
        elif axis == 1:
            first = (frame[:,0] == 1).any()
            last = (frame[:,len(frame[:]) - 1] == 1).any()
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
        
        
    first_frame = binarize(image[0], R_thresh)
    last_frame = binarize(image[-1], R_thresh)
    return (check_connected(first_frame) and check_connected(last_frame)) or (check_connected(first_frame, axis = 1) and check_connected(last_frame, axis = 1))

def track_void(image, threshold, step):
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
    
    void_lst = []
    island_area_lst = []
    island_position_lst = []
    
    for i in range(0, len(image), step):
        new_frame = binarize(image[i], threshold)
        void_area = find_largest_void(new_frame)
        island_area = find_largest_void(new_frame, find_void = False)
        island_position = largest_island_position(new_frame)
        void_lst.append(void_area)
        island_area_lst.append(island_area)
        island_position_lst.append(island_position)
    return void_lst, island_area_lst, island_position_lst

def check_resilience(file, channel, R_offset, percent_threshold_loss, percent_threshold_gain, frame_step, frame_start_percent, frame_stop_percent):
    #Note for parameters: frame_step (stepsize) used to reduce the runtime. 
    image = file[:,:,:,channel]
    frame_initial_percent = 0.05

    fig, ax = plt.subplots(figsize = (5,5))

    # Error Checking: Empty Image
    if (image == 0).all():
        verdict = "Data not available for this channel."
        return [None] * 7
    
    largest_void_lst, island_area_lst, island_position_lst = track_void(image, R_offset, frame_step)
    start_index = int(len(largest_void_lst) * frame_start_percent)
    stop_index = int(len(largest_void_lst) * frame_stop_percent)
    start_initial_index = int(len(largest_void_lst)*frame_initial_percent)

    percent_gain_initial_list = np.mean(largest_void_lst[0:start_initial_index])
    percent_gain_list = np.array(largest_void_lst)/percent_gain_initial_list
    
    ax.plot(np.arange(start_index, stop_index), percent_gain_list[start_index:stop_index])
    ax.set_xlabel("Frames")
    ax.set_ylabel("Proportion of orginal void size")
    #Calculate
    
    avg_percent_change = np.mean(largest_void_lst[start_index:stop_index])/percent_gain_initial_list
    max_void_size = max(largest_void_lst)/(len(image[0,0,:])*len(image[0,:,0]))
    island_size = max(island_area_lst)/(len(image[0,0,:])*len(image[0,:,0]))
    island_movement = np.array(island_position_lst)[:-1,:] - np.array(island_position_lst)[1:,:]
    island_speed = np.linalg.norm(island_movement,axis = 1)
    island_direction = np.arctan2(island_movement[:,1],island_movement[:,0])
    island_direction = island_direction[np.where(island_speed < 15)]
    average_direction = np.average(island_direction)
    #Give judgement
    if avg_percent_change >= percent_threshold_loss and avg_percent_change <= percent_threshold_gain or max_void_size < 0.10:
        verdict = 1
    else:
        verdict = 0
    
    spanning = check_span(image, R_offset)
    
    return verdict, fig, max_void_size, spanning, island_size, average_direction, avg_percent_change

def main():
    file = read_file(sys.argv[1])
    channel = read_file(sys.argv[2])
    verdict, fig, void_value, spanning, island_size, average_direction, avg_percent_change = check_resilience(file, channel)

if __name__ == "__main__":
    main()
    
