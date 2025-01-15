import numpy as np
import matplotlib.pyplot as plt
import csv, os, functools, builtins
from scipy.stats import mode, kurtosis

def calc_mode(frame):
    mode_result = mode(frame.flatten(), keepdims=False)
    mode_intensity = mode_result.mode if isinstance(mode_result.mode, np.ndarray) else np.array([mode_result.mode])
    mode_intensity = mode_intensity[0] if mode_intensity.size > 0 else np.nan
    return mode_intensity

def calc_mode_skewness(frame):
    mean_intensity = np.mean(frame)
    mode_intensity = calc_mode(frame)
    stdev_intensity = np.std(frame)
    return (mean_intensity - mode_intensity)/stdev_intensity

def calc_median_skewness(frame):
    mean_intensity = np.mean(frame)
    median_intensity = np.median(frame)
    stdev_intensity = np.std(frame)
    return 3 * (mean_intensity - median_intensity)/stdev_intensity

def top_ten_average(lst):
    lst.sort(reverse=True)
    length = len(lst)
    top_ten_percent = int(np.ceil(length * 0.1))
    return np.mean(lst[0:top_ten_percent])

def populate_intensity_array(video, first, last, ignore_saturation = True):
    i_arr = []
    dropped_frames = 0
    inc = 1 if first < last else -1
    frame_no = first
    while len(i_arr) < np.abs(last - first):
        if frame_no >= len(video) or frame_no < 0:
            return None, None, None
        frame = video[frame_no]
        f_mode = calc_mode(frame)
        f_max = np.max(frame)
        while f_max == f_mode:
            frame_no += inc
            if (frame_no >= len(video) or frame_no < 0) and not ignore_saturation:
                return None, None, None
            elif (frame_no >= len(video) or frame_no < 0):
                i_arr = [video[i] for i in range(first, last, inc)]
                dropped_frames = len(video)
                frame_no = last
                return np.array(i_arr), dropped_frames, frame_no
            frame = video[frame_no]
            f_mode = calc_mode(frame)
            f_max = np.max(frame)
            dropped_frames += 1
        i_arr.append(frame)
        frame_no += inc
    return np.array(i_arr), dropped_frames, frame_no

def calc_frame_metric(metric, data):
    mets = []
    for i in range(len(data)):
        met = metric(data[i].flatten())
        mets.append(met)
    return mets

def check_coarse(file, name, channel, first_frame, last_frame, frames_percent, save_intermediates, verbose):
    flag = 0 # No flags have been tripped by the module
    print = functools.partial(builtins.print, flush=True)
    vprint = print if verbose else lambda *a, **k: None
    vprint('Beginning Coarsening Testing')

    im = file[:,:,:,channel]

    num_frames = im.shape[0]
    num_frames_analysis = int(np.ceil(frames_percent * num_frames))

    # Set last_frame to last frame of movie if unspecified

    fig, ax = plt.subplots(figsize=(5,5))

    if (im == 0).all(): # If image is blank, then end program early
        return np.nan, [np.nan] * 6, np.nan
    
    max_px_intensity = 1.1*np.max(im)
    bins_width = 3

    i_frames_data = [im[i] for i in range(0, num_frames_analysis, 1)]
    f_frames_data = [im[i] for i in range(last_frame - num_frames_analysis, last_frame, 1)]

    # Check for saturation, posts saturation flag (flag = 2) if mode and maximal intensity values are same
    if [np.max(frame) == calc_mode(frame) for frame in i_frames_data + f_frames_data].all():
        flag == 2

    if save_intermediates:
        filename = os.path.join(name, 'IntensityDistribution.csv')
        with open(filename, "w") as myfile:
            csvwriter = csv.writer(myfile)
            for frame_idx in range(0, num_frames_analysis, 1):
                csvwriter.writerow(['Frame ' + str(frame_idx)])
                frame_data = im[frame_idx]
                frame_values, frame_counts = np.unique(frame_data, return_counts = True)
                csvwriter.writerow(frame_values)
                csvwriter.writerow(frame_counts)
                csvwriter.writerow([])

            for frame_idx in range(last_frame, last_frame - num_frames_analysis, -1):
                csvwriter.writerow(['Frame ' + str(frame_idx)])
                frame_data = im[frame_idx]
                frame_values, frame_counts = np.unique(frame_data, return_counts = True)
                csvwriter.writerow(frame_values)
                csvwriter.writerow(frame_counts)
                csvwriter.writerow([])
    
    # Use the calc_frame_metric function to get the kurtosis, skew, and asymmetry for each of the first and last 10% of frames
    i_kurt = calc_frame_metric(kurtosis, i_frames_data)
    f_kurt = calc_frame_metric(kurtosis, f_frames_data)
    tot_kurt = i_kurt + f_kurt
    i_median_skew = calc_frame_metric(calc_median_skewness, i_frames_data)
    f_median_skew = calc_frame_metric(calc_median_skewness, f_frames_data)
    tot_median_skew = i_median_skew + f_median_skew
    i_mode_skew = calc_frame_metric(calc_mode_skewness, i_frames_data)
    f_mode_skew = calc_frame_metric(calc_mode_skewness, f_frames_data)
    tot_mode_skew = i_mode_skew + f_mode_skew

    # Take the largest ten percent of each metric and average them
    max_kurt = top_ten_average(tot_kurt)
    max_median_skew = top_ten_average(tot_median_skew)
    max_mode_skew = top_ten_average(tot_mode_skew)

    # Take the difference of the average between the first and last 10% of frames
    kurt_diff = np.mean(np.array(f_kurt)) - np.mean(np.array(i_kurt))
    median_skew_diff = np.mean(np.array(f_median_skew)) - np.mean(np.array(i_median_skew))
    mode_skew_diff = np.mean(np.array(f_mode_skew)) - np.mean(np.array(i_mode_skew))

    # Plot the intensity distributions for the first and last frame for comparison
    i_frame = im[first_frame]
    f_frame = im[last_frame] if (last_frame != False and last_frame < len(im)) else im[-1]
    
    fig, ax = plt.subplots(figsize=(5,5))
    set_bins = np.arange(0, max_px_intensity, bins_width)
    i_count, bins = np.histogram(i_frame.flatten(), bins=set_bins, density=True)
    f_count, bins = np.histogram(f_frame.flatten(), bins=set_bins, density=True)
    center_bins = (bins[1] - bins[0])/2
    plt_bins = bins[0:-1] + center_bins
        
    i_mean = np.mean(i_frame)
    f_mean = np.mean(f_frame)
    
    ax.plot(plt_bins[::10], i_count[::10], '^-', ms=4, c='darkred', alpha=0.6, label= "Frame " + str(first_frame)+" Intensity Distribution")
    ax.plot(plt_bins[::10], f_count[::10], 'v-', ms=4, c='purple',   alpha=0.6, label= "Frame " + str(last_frame)+" Intensity Distribution")
    ax.axvline(x=i_mean, ms = 4, c = 'darkred', alpha=1, label="Frame " + str(first_frame)+" Mean")
    ax.axvline(x=f_mean, ms = 4, c = 'purple', alpha=1, label="Frame " + str(last_frame)+" Mean")

    ax.axhline(0, color='dimgray', alpha=0.6)
    ax.set_xlabel("Pixel intensity value")
    ax.set_ylabel("Probability")
    ax.set_yscale('log')
    ax.set_xlim(0,max_px_intensity)
    ax.legend()
    
    return fig, [max_kurt, max_median_skew, max_mode_skew, kurt_diff, median_skew_diff, mode_skew_diff], flag