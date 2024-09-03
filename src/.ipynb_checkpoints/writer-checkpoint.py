import os, csv, functools, builtins
import matplotlib.pyplot as plt
import numpy as np

def write_file(output_filepath, data):
    if data:
        headers = ['Channel', 'Flags', 'Connectivity', 'Largest Island Size', 'Largest Void Size', 'Void Size Change', 'Maximum Kurtosis', 'Maximum Skewness', 'Maximum Mean-Mode', 'Mean-Mode Difference', 'Kurtosis Difference', 'Skewness Difference', 'Mean Velocity', 'Mean Speed', 'Mean Divergence', 'Island Movement Direction', 'Mean Flow Direction', 'Flow Direction (Standard Deviation)']
        with open(output_filepath, 'w', newline='', encoding="utf-8") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(headers) # Write headers before the first filename
            headers = [] # Ensures headers are only written once per file
            for entry in data:
                if isinstance(entry, list) and len(entry) == 1:
                    # Write the file name
                    csvwriter.writerow(entry)
                    # csvwriter.writerow(headers)  # Write headers after the filename
                elif entry:
                    csvwriter.writerow(entry)
                else:
                    # Write an empty row
                    csvwriter.writerow([])

def generate_aggregate_csv(filelist, csv_loc, gen_agg_barcode, normalize):
    if gen_agg_barcode:
        combined_barcode_loc = os.path.join(os.path.dirname(csv_loc), 'aggregate_barcode')
    headers = ['Channel', 'Flags', 'Connectivity', 'Largest Island Size', 'Largest Void Size', 'Void Size Change', 'Maximum Kurtosis', 'Maximum Skewness', 'Maximum Mean-Mode', 'Mean-Mode Difference', 'Kurtosis Difference', 'Skewness Difference', 'Mean Velocity', 'Mean Speed', 'Mean Divergence', 'Island Movement Direction', 'Mean Flow Direction', 'Flow Direction (Standard Deviation)']
    f = open(csv_loc, 'w', encoding="utf-8") # Clears the CSV file if it already exists, and creates it if it does not
    csv_writer = csv.writer(f)
    csv_writer.writerow(headers)
    f.close()
    
    def combine_csvs(csv_list):
        filenames = []
        csv_data = np.zeros(shape=(18))
        if not csv_list:
            return None
        for csv_file in csv_list:
            with open(csv_file, 'r') as fread, open(combined_csv_loc, 'a', encoding="utf-8") as fwrite:
                csv_reader = csv.reader(fread)
                csv_writer = csv.writer(fwrite)
                next(csv_reader, None)
                for row in csv_reader:
                    if len(row) == 1:
                        filenames.append(str(row))
                    elif len(row) == 0:
                        continue
                    else:
                        row = np.float_(row)
                        arr_row = np.array(row)
                        csv_data = np.vstack((csv_data, arr_row))
                    csv_writer.writerow(row)
        return csv_data

    csv_data = combine_csvs(files)
    
    if gen_barcode:
        csv_data_2 = csv_data[1:]
        gen_combined_barcode(csv_data_2, combined_barcode_loc, normalize_data)

def gen_combined_barcode(data, figpath, normalize_data = True):
    channels = data[:,0]
    unique_channels = np.unique(channels)
    flags = data[:,1]
    connectivity = data[:,2]
    island_size = data[:,3]
    void_value = data[:,4]
    void_growth = data[:,5]
    max_kurtosis = data[:,6]
    max_skewness = data[:,7]
    max_mean_mode = data[:,8]
    mean_mode_diff = data[:,9]
    kurtosis_diff = data[:,10]
    skewness_diff = data[:,11]
    avg_vel = data[:,12]
    avg_speed = data[:,13]
    avg_div = data[:,14]
    island_dir = data[:,15]
    flow_dir = data[:,16]
    flow_dir_sd = data[:,17]
    all_entries = [connectivity, island_size, void_value, void_growth, max_kurtosis, max_skewness, max_mean_mode, mean_mode_diff, kurtosis_diff, skewness_diff, avg_vel, avg_speed, avg_div, island_dir, flow_dir, flow_dir_sd]

    # Define normalization limits of floating point values
    connected_lim = [0, 1] # Limit on the percentage of frames that are connected
    bin_size_lim = [0, 1]  # Size limit for void and island size
    void_growth_lim = [0, 5] # Limits for the expected growth of the void
    c_lim = [0, 10] # Limit on the percentage of mean-mode difference thresholding
    kurt_lim = [-10, 10] # Limit on the kurtosis
    skew_lim = [-10, 10] # Limit on the skewness
    avg_vel_lim = [0, 10] # Limit for the average velocity (pixels/sec)
    avg_speed_lim = [0, 10] # Limit for the average speed (pixels/sec)
    avg_div = [-1, 1] # Limit for divergence metric (-1 is pure contraction, 1 is pure expansion)
    direct_lim = [-np.pi, np.pi] # Limits on the direction of the island and flow (radians)
    
    limits = [connected_lim, bin_size_lim, bin_size_lim, void_growth_lim, kurt_lim, skew_lim, c_lim, c_lim, kurt_lim, skew_lim, avg_vel_lim, avg_speed_lim, avg_div, direct_lim, direct_lim, direct_lim]

    if normalize_data:
        for i in range(16):
            limits[i] = [np.min(all_entries[i]), np.max(all_entries[i])]
    colormap = plt.get_cmap('plasma')  # Colormap for floats

    def normalize(x, min_float, max_float):
        if x == None:
            return None
        return (x - min_float) / (max_float - min_float)

    for channel in unique_channels:
        channel_figpath = figpath + '_channel_' + str(int(channel)) + '.png'
        filtered_channel_data = data[data[:,0] == channel]
        channel_agg_barcode = [None] * len(filtered_channel_data)
        for row in range(len(filtered_channel_data)):
            barcode = [None] * 16
            for idx in range(len(all_entries)):
                value = filtered_channel_data[row, 2 + idx]
                lims = limits[idx]
                cval = normalize(value, lims[0], lims[1])
                color = [0.5, 0.5, 0.5] if cval == None else colormap(cval)[:3]
                barcode[idx] = color
            channel_agg_barcode[row] = barcode
            
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
    
        # Repeat each barcode to make it more visible
        barcode_image = np.repeat(channel_agg_barcode, 5, axis=0)  # Adjust the repetition factor as needed
    
        # Plot the stitched barcodes
        ax.imshow(channel_agg_barcode, aspect='auto')
        ax.axis('off')  # Turn off the axis
        
        # Save or show the figure
        plt.savefig(channel_figpath, bbox_inches='tight', pad_inches=0)
