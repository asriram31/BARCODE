import os, csv, functools, builtins
import matplotlib.pyplot as plt
import numpy as np

def write_file(output_filepath, data):
    if data:
        headers = [
            'Channel', 'Flags', 'Connectivity', 'Maximum Island Area', 'Maximum Void Area', 
            'Void Area Change', 'Island Area Change', 'Initial Island Area 1', 
            'Initial Island Area 2', 'Maximum Kurtosis', 'Maximum Median Skewness', 
            'Maximum Mode Skewness', 'Kurtosis Difference', 'Median Skewness Difference', 
            'Mode Skewness Difference', 'Mean Speed', 'Speed Change',
            'Mean Flow Direction', 'Flow Directional Spread']
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

def generate_aggregate_csv(filelist, csv_loc, gen_barcode, normalize, num_params = 19):
    if gen_barcode:
        combined_barcode_loc = os.path.join(os.path.dirname(csv_loc), 'aggregate_barcode')
        headers = [
            'Channel', 'Flags', 'Connectivity', 'Maximum Island Area', 'Maximum Void Area', 
            'Void Area Change', 'Island Area Change', 'Initial Island Area 1', 
            'Initial Island Area 2', 'Maximum Kurtosis', 'Maximum Median Skewness', 
            'Maximum Mode Skewness', 'Kurtosis Difference', 'Median Skewness Difference', 
            'Mode Skewness Difference', 'Mean Speed', 'Speed Change',
            'Mean Flow Direction', 'Flow Directional Spread']
    f = open(csv_loc, 'w', encoding="utf-8") # Clears the CSV file if it already exists, and creates it if it does not
    csv_writer = csv.writer(f)
    csv_writer.writerow(headers)
    f.close()
    
    def combine_csvs(csv_list):
        filenames = []
        csv_data = np.zeros(shape=(num_params))
        if not csv_list:
            return None
        for csv_file in csv_list:
            with open(csv_file, 'r') as fread, open(csv_loc, 'a', encoding="utf-8") as fwrite:
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

    csv_data = combine_csvs(filelist)
    
    if gen_barcode:
        csv_data_2 = csv_data[1:]
        gen_combined_barcode(csv_data_2, combined_barcode_loc, normalize)

def gen_combined_barcode(data, figpath, normalize_data = True, num_params = 19):
    if len(data.shape) <= 1:
        data = np.reshape(data, (1, data.shape[0]))
    if data.shape[1] == 0:
        return
    channels = data[:,0]
    unique_channels = np.unique(channels)
    flags = data[:,1]
    all_entries = [data[:,i] for i in range(2, num_params)]

    # Define normalization limits of floating point values
    connected_lim = [0, 1] # Limit on the percentage of frames that are connected
    bin_size_lim = [0, 1]  # Size limit for void and island size
    bin_growth_lim = [0, 5] # Limits for the expected growth of the void or island
    c_lim = [0, 10] # Limit on the percentage of asymmetry difference thresholding
    kurt_lim = [-10, 10] # Limit on the kurtosis
    skew_lim = [-10, 10] # Limit on the skewness
    avg_speed_lim = [0, 10] # Limit for the average speed (pixels/sec)
    diff_speed_lim = [0, 10] # Limit for speed difference
    direct_lim = [-np.pi, np.pi] # Limits on the direction of the island and flow (radians)
    
    limits = [connected_lim, bin_size_lim, bin_size_lim, bin_growth_lim, bin_growth_lim, bin_size_lim, bin_size_lim, kurt_lim, skew_lim, c_lim, kurt_lim, skew_lim, c_lim, avg_speed_lim, diff_speed_lim, direct_lim, direct_lim]

    if normalize_data:
        for i in range(num_params - 2):
            if all(v is None for v in all_entries[i]):
                continue
            # entries = [x for x in all_entries[i] if (x != np.nan and x != None)]
            limits[i] = [np.nanmin(all_entries[i]), np.nanmax(all_entries[i])]
    colormap = plt.get_cmap('plasma')  # Colormap for floats

    def normalize(x, min_float, max_float):
        if x == None:
            return None
        return (x - min_float) / (max_float - min_float)

    for channel in unique_channels:
        channel_figpath = f'{figpath}_channel_{channel}.png'
        filtered_channel_data = data[data[:,0] == channel]
        channel_agg_barcode = [None] * len(filtered_channel_data)
        for row in range(len(filtered_channel_data)):
            barcode = [None] * (num_params - 2)
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
        ax.imshow(barcode_image, aspect='auto')
        ax.axis('off')  # Turn off the axis
        
        # Save or show the figure
        fig.savefig(channel_figpath, bbox_inches='tight', pad_inches=0)
