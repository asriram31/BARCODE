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

class MyException(Exception):
    pass

def inv(arr):
    ones_arr = np.ones(shape = arr.shape)
    return ones_arr - arr

def groupAvg(arr, N, bin_mask=True):
    result = np.cumsum(arr, 0)[N-1::N]/float(N)
    result = np.cumsum(result, 1)[:,N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]
    result[:,1:] = result[:,1:] - result[:,:-1]
    if bin_mask:
        result = np.where(result > 0, 1, 0)
    return result

def binarize(frame, offset_threshold):
    avg_intensity = np.mean(frame)
    threshold = avg_intensity * (1 + offset_threshold)
    new_frame = np.where(frame < threshold, 0, 1)
    return new_frame

def top_ten_average(lst):
    lst.sort(reverse=True)
    length = len(lst)
    top_ten_percent = int(np.ceil(length * 0.1))
    return np.mean(lst[0:top_ten_percent])

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

def track_void(image, name, threshold, step, return_graphs, save_intermediates):
    def find_largest_void(frame, find_void, num=1):      
        if find_void:
            eval_frame = inv(frame)
        else:
            eval_frame = frame
        labeled, a = label(eval_frame, connectivity= 2, return_num =True) # identify the regions of connectivity 2
        if a == 0:
            return frame.shape[0] * frame.shape[1]
        
        regions = regionprops(labeled) # determines the region properties of the labeled
        if not regions:
            return frame.shape[0] * frame.shape[1]
        
        regions_sorted = sorted(regions, key = lambda r: r.area, reverse = True)
        largest_regions = regions_sorted[0:num]
        areas = [region.area for region in largest_regions]
        if num != len(areas):
            areas.append(0)
        return areas # returns largest region(s) area
    


    def largest_island_position(frame):      
        labeled, a = label(frame, connectivity = 2, return_num =True) # identify the regions of connectivity 2
        if a == 0:
            return None
        regions = regionprops(labeled) # determines the region properties of the labeled
        largest_region = max(regions, key = lambda r: r.area) # determines the region with the maximum area
        return largest_region.centroid # returns largest region area
        
    if save_intermediates:
        filename = os.path.join(name, 'BinarizationData.csv')
        f = open(filename, 'w')
        csvwriter = csv.writer(f)
        
    void_lst = []
    island_area_lst = []
    island_area_lst2 = []
    island_position_lst = []
    connected_lst = []
    region_lst = []
    
    mid_point_arr = range(0, len(image), step)
    mid_point = mid_point_arr[int((len(mid_point_arr) - 1)/2)]
    save_spots = np.array([0, mid_point, len(image)])

    for i in range(0, len(image), step):
        new_image = binarize(image[i], threshold)
        new_frame = groupAvg(new_image, 2)
        
        if i in save_spots and return_graphs:
            compare_fig, comp_axs = plt.subplots(ncols = 2, figsize=(10, 5))
            comp_axs[0].imshow(image[i], cmap='gray')
            comp_axs[1].imshow(new_frame, cmap='gray')
            ticks_adj = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 2))
            comp_axs[1].xaxis.set_major_formatter(ticks_adj)
            comp_axs[1].yaxis.set_major_formatter(ticks_adj)
            comp_axs[0].axis('off')  # Turn off the axis
            comp_axs[1].axis('off')  # Turn off the axis
            plt.savefig(os.path.join(name, 'Binarization Frame ' + str(i) + ' Comparison.png'))
            plt.close('all')
            
        if save_intermediates:
            csvwriter.writerow([str(i)])
            csvwriter.writerows(new_frame)
            csvwriter.writerow([])

        island_area_lst.append(find_largest_void(new_frame, find_void = False)[0])
        island_area_lst2.append(find_largest_void(new_frame, find_void = False, num = 2)[1])
        island_position_lst.append(largest_island_position(new_frame))
        connected_lst.append(check_span(new_frame))
        void_lst.append(find_largest_void(new_frame, True)[0])
        
        labeled, a = label(new_frame, connectivity = 2, return_num =True) # identify the regions of connectivity 2

        regions = regionprops(labeled) # determines the region properties of the labeled
        
        region_lst.append(regions)
    i = len(image) - 1    
    if i % step != 0:
        new_image = binarize(image[i], threshold)
        new_frame = groupAvg(new_image, 2)
        
        if i in save_spots and return_graphs:
            compare_fig, comp_axs = plt.subplots(2, figsize=(10, 5))
            comp_axs[0].imshow(new_image)
            comp_axs[1].imshow(new_frame)
            plt.savefig(os.path.join(name, 'Binarization Frame ' + str(i) + ' Comparison.png'))
        
        if save_intermediates:
            csvwriter.writerow([str(i)])
            csvwriter.writerows(new_frame)
            csvwriter.writerow([])
        
        island_area_lst.append(find_largest_void(new_frame, find_void = False)[0])
        island_area_lst2.append(find_largest_void(new_frame, find_void = False, num = 2)[1])
        island_position_lst.append(largest_island_position(new_frame))
        connected_lst.append(check_span(new_frame))
        void_lst.append(find_largest_void(new_frame, True)[0])

    if save_intermediates:
        f.close()

    return void_lst, island_area_lst, island_area_lst2, connected_lst

def check_resilience(file, name, channel, R_offset = 0.1, frame_step = 10, frame_start_percent = 0.9, frame_stop_percent = 1, return_graphs = False, save_intermediates = False, verbose = True):
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
        frame_step = int(frame_step / 5)
    
    largest_void_lst, island_area_lst, island_area_lst2, connected_lst = track_void(image, name, R_offset, frame_step, return_graphs, save_intermediates)
    start_index = int(np.floor(len(image) * frame_start_percent / frame_step))
    stop_index = int(np.ceil(len(largest_void_lst) * frame_stop_percent))
    start_initial_index = int(np.ceil(len(image)*frame_initial_percent / frame_step))

    void_size_initial = np.mean(largest_void_lst[0:start_initial_index])
    void_percent_gain_list = np.array(largest_void_lst)/void_size_initial
    
    island_size_initial = np.mean(island_area_lst[0:start_initial_index])
    island_size_initial2 = np.mean(island_area_lst2[0:start_initial_index])
    island_percent_gain_list = np.array(island_area_lst)/island_size_initial
    
    start_index = 0
    plot_range = np.arange(start_index * frame_step, stop_index * frame_step, frame_step)
    plot_range[-1] = len(image) - 1 if stop_index * frame_step >= len(image) else stop_index * frame_step
    ax.plot(plot_range, 100 * void_percent_gain_list[start_index:stop_index], c='b', label='Original Void Size Proportion')
    ax.plot(plot_range, 100 * island_percent_gain_list[start_index:stop_index], c='r', label='Original Island Size Proportion')
    ax.set_xticks(plot_range[::10])
    if stop_index * frame_step >= len(image) != 0:
        ax.set_xlim(left=None, right=len(image) - 1)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Percentage of Original Size")
    ax.legend()

    downsample = 2
    
    img_dims = image[0].shape[0] * image[0].shape[1] / (downsample ** 2)
    
    avg_void_percent_change = np.mean(largest_void_lst[start_index:stop_index])/void_size_initial
    void_size_initial = void_size_initial / img_dims
    max_void_size = top_ten_average(largest_void_lst)/img_dims
    
    avg_island_percent_change = np.mean(island_area_lst[start_index:stop_index])/island_size_initial
    island_size_initial = island_size_initial / img_dims
    island_size_initial2 = island_size_initial2 / img_dims
    max_island_size = top_ten_average(island_area_lst)/img_dims
        
    spanning = len([con for con in connected_lst if con == 1])/len(connected_lst)

    return fig, [spanning, max_island_size, max_void_size, avg_void_percent_change, avg_island_percent_change, island_size_initial, island_size_initial2]
