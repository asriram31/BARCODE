import csv, functools, builtins
import matplotlib.pyplot as plt
import numpy as np

def write_file(output_filepath, data):
    if data:
        headers = ['Channel', 'Connectivity', 'Island Size', 'Largest Void', 'Void Size Change', 'Coarsening', 'Intensity Difference Area 1', 'Intensity Difference Area 2', 'Kurtosis', 'Skewness', 'Mean Velocity', 'Mean Speed', 'Mean Divergence', 'Island Movement Direction', 'Mean Flow Direction', 'Flow Direction (Standard Deviation)']
        with open(output_filepath, 'w', newline='') as csvfile:
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

def create_barcode(entry):
    # Define color mappings
    colormap = plt.get_cmap('plasma')  # Colormap for floats

    # channel, r, c, spanning, void_value, island_size, island_movement, void_growth, direct, avg_vel, avg_speed, avg_div, c_area1, c_area2 = entry
    channel = entry[0]
    float_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    float_values = [entry[val] for val in float_indices]

    # Define normalization limits of floating point values
    connected_lim = [0, 1] # Limit on the percentage of frames that are connected
    bin_size_lim = [0, 1]  # Size limit for void and island size
    void_growth_lim = [0, 5] # Limits for the expected growth of the void
    c_lim = [0, 10] # Limit on the percentage of mean-mode difference thresholding
    i_area_lim = [0, 1] # Limit for the first two areas of the delta-I distribution
    kurt_lim = [0, 10]
    skew_lim = [0, 10]
    avg_vel_lim = [0, 10] # Limit for the average velocity (pixels/sec)
    avg_speed_lim = [0, 10] # Limit for the average speed (pixels/sec)
    avg_div = [-1, 1] # Limit for divergence metric (-1 is pure contraction, 1 is pure expansion)
    direct_lim = [-np.pi, np.pi] # Limits on the direction of the island and flow (radians)
    
    limits = [connected_lim, bin_size_lim, bin_size_lim, void_growth_lim, c_lim, i_area_lim, i_area_lim, kurt_lim, skew_lim, avg_vel_lim, avg_speed_lim, avg_div, direct_lim, direct_lim, direct_lim]
    
    def normalize(x, min_float, max_float):
        if x == None:
            return None
        return (x - min_float) / (max_float - min_float)
    
    # Create the color barcode
    barcode = [None] * 15

    for f_index, entry_value_float, float_lim in zip(float_indices, float_values, limits):
        cval = normalize(entry_value_float, float_lim[0], float_lim[1])
        color = [0.5, 0.5, 0.5] if cval == None else colormap(cval)[:3]
        barcode[f_index - 1] = color
    
    # Convert to numpy array and reshape for plotting
    barcode = np.array(barcode)

    return barcode
        


def generate_stitched_barcode(barcodes, figpath=None):
    # Stack the barcodes vertically
    stitched_barcodes = np.vstack(barcodes)
    
    # Normalize the RGB values across all barcodes using the plasma colormap
    num_barcodes = len(barcodes)
    plasma_colormap = plt.get_cmap('plasma')
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Repeat each barcode to make it more visible
    barcode_image = np.repeat(stitched_barcodes, 10, axis=0)  # Adjust the repetition factor as needed
    
    # Plot the stitched barcodes
    ax.imshow(barcode_image, aspect='auto')
    ax.axis('off')  # Turn off the axis
    
    # Save or show the figure
    plt.savefig(figpath, bbox_inches='tight', pad_inches=0)