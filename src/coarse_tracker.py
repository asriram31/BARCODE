import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
# from numpy.polynomial import Polynomial, polyroots
from nd2reader import ND2Reader
from scipy.interpolate import splrep, sproot, BSpline
import scipy, csv, os, functools, builtins
from scipy.stats import mode, skew, kurtosis
import scipy.signal as signal

def calculate_mean_mode(frame):
    mean_intensity = np.mean(frame)
    mode_result = mode(frame.flatten(), keepdims=False)
    mode_intensity = mode_result.mode if isinstance(mode_result.mode, np.ndarray) else np.array([mode_result.mode])
    mode_intensity = mode_intensity[0] if mode_intensity.size > 0 else np.nan
    return mean_intensity, mode_intensity

def analyze_frames(name, video, frames_percent, save_intermediates):
    num_frames = video.shape[0]
    num_frames_analysis = int(np.ceil(frames_percent * num_frames))
    def get_mean_mode_diffs(frames):
        diffs = []
        means = []
        modes = []
            
        for frame in frames:
            mean_intensity, mode_intensity = calculate_mean_mode(frame)
            diffs.append(mean_intensity - mode_intensity)
            means.append(mean_intensity)
            modes.append(mode_intensity)
            
        avg_diffs = np.mean(diffs)
        avg_means = np.mean(means)
        avg_modes = np.mean(modes)
            
        return avg_diffs, avg_means, avg_modes
        
    # Get the first five frames and the last five frames
    first_frames = [video[i] for i in range(num_frames_analysis)]
    last_frames = [video[num_frames - 1 - i] for i in range(num_frames_analysis)]

    if save_intermediates:
        filename = os.path.join(name, 'IntensityDistribution.csv')
        with open(filename, "w") as myfile:
            csvwriter = csv.writer(myfile)
            for frame_idx in range(0, num_frames_analysis, 1):
                csvwriter.writerow(['Frame ' + str(frame_idx)])
                frame_data = video[frame_idx]
                frame_values, frame_counts = np.unique(frame_data, return_counts = True)
                csvwriter.writerow(frame_values)
                csvwriter.writerow(frame_counts)
                csvwriter.writerow([])

            for frame_idx in range(num_frames_analysis, 0, -1):
                csvwriter.writerow(['Frame ' + str(num_frames - 1 - frame_idx)])
                frame_data = video[num_frames - 1 - frame_idx]
                frame_values, frame_counts = np.unique(frame_data, return_counts = True)
                csvwriter.writerow(frame_values)
                csvwriter.writerow(frame_counts)
                csvwriter.writerow([])
            
        
    # Calculate the average difference, mean, and mode for the first and last five frames for each channel
    avg_diff_first, avg_means_first, avg_modes_first = get_mean_mode_diffs(first_frames)
    avg_diff_last, avg_means_last, avg_modes_last = get_mean_mode_diffs(last_frames)
        
    # Calculate the percentage increase for each channel
    percentage_increase = (avg_diff_last - avg_diff_first)/avg_diff_first * 100 if avg_diff_first != 0 else np.nan
    
    # Check if any channel meets the "coarsening" criteria
    return percentage_increase

def check_coarse(file, name, channel, first_frame, last_frame, frames_percent, save_intermediates, verbose):
    print = functools.partial(builtins.print, flush=True)
    vprint = print if verbose else lambda *a, **k: None
    vprint('Beginning Coarsening Testing')
    extrema_bounds_list = []
    extrema_bounds_idx_list = []
    areas_list = []
    extrema_len_list = []
    extrema_height_list = []

    im = file[:,:,:,channel]

    num_frames = im.shape[0]
    num_frames_analysis = int(np.ceil(frames_percent * num_frames))

    # Set last_frame to last frame of movie if unspecified

    fig, ax = plt.subplots(figsize=(5,5))

    if (im == 0).all(): # If image is blank, then end program early
        return [None] * 6

    max_px_intensity = 1.1*np.max(im)
    min_px_intensity = np.min(im)
    bins_width = 3
    poly_deg = 40
    poly_len = 10000
    
    near_zero_limit = 0.01
    minimum_area = 0.010

    
    i_frames_data = np.array([im[i] for i in range(first_frame, first_frame+num_frames_analysis)])
    if last_frame == False or last_frame >= len(im): 
        f_frames_data = np.array([im[num_frames - 1 - i] for i in range(num_frames_analysis)])

    else:
        f_frames_data = np.array([im[last_frame - 1 - i] for i in range(num_frames_analysis)])

    i_kurt = kurtosis(i_frames_data.flatten())
    f_kurt = kurtosis(f_frames_data.flatten())
    i_skew = skew(i_frames_data.flatten())
    f_skew = skew(f_frames_data.flatten())

    kurt_diff = f_kurt - i_kurt
    skew_diff = f_skew - i_skew

    fig, ax = plt.subplots(figsize=(5,5))
    set_bins = np.arange(0, max_px_intensity, bins_width)
    bins_num = len(set_bins)
    i_count, bins = np.histogram(i_frames_data.flatten(), bins=set_bins, density=True)
    f_count, bins = np.histogram(f_frames_data.flatten(), bins=set_bins, density=True)
    center_bins = (bins[1] - bins[0])/2
    plt_bins = bins[0:-1] + center_bins
    ax.plot(plt_bins, i_count, '^-', ms=4, c='darkred', alpha=0.2, label= "frame " + str(first_frame+1)+" dist")
    ax.plot(plt_bins, f_count, 'v-', ms=4, c='darkorange',   alpha=0.2, label= "frame " + str(last_frame+1)+" dist")
    
    count_diff = f_count - i_count
    ax.plot(plt_bins, count_diff, 'D-', ms=2, c='red', label = "difference btwn")
    
    ax.plot(plt_bins, i_count, '^-', ms=4, c='darkred', alpha=0.2, label= "frame " + str(first_frame+1)+" dist")
    ax.plot(plt_bins, f_count, 'v-', ms=4, c='darkorange',   alpha=0.2, label= "frame " + str(last_frame+1)+" dist")
    count_diff = f_count - i_count
    ax.plot(plt_bins, count_diff, 'D-', ms=2, c='red', label = "difference btwn")
    
    
    # ### get range for local extrema of interest ###

    cumulative_count_diff = np.cumsum(count_diff)
    filtered_ccd = scipy.ndimage.gaussian_filter1d(cumulative_count_diff, 8)
    
    peaks_max = signal.argrelextrema(filtered_ccd, np.greater, order = 20)
    peaks_min = signal.argrelextrema(filtered_ccd, np.less, order = 20)
    filt_max = np.array([0]) if len(filtered_ccd[peaks_max]) == 0 else filtered_ccd[peaks_max][0]
    filt_min = np.array([0]) if len(filtered_ccd[peaks_min]) == 0 else filtered_ccd[peaks_min][0]
    areas = np.append(np.abs(filt_max), np.abs(filt_min))

    perc_increase = analyze_frames(name, im, frames_percent, save_intermediates)

    ax.axhline(0, color='dimgray', alpha=0.6)
    ax.set_xlabel("Pixel intensity value")
    ax.set_ylabel("Probability")
    ax.set_xlim(0,max_px_intensity + 5)
    ax.legend()
    
    return perc_increase, fig, areas[0], areas[1], kurt_diff, skew_diff