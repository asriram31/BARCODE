import csv, functools, builtins
import matplotlib.pyplot as plt
import numpy as np

def write_file(output_filepath, data):
    if data:
        headers = ['Channel', 'Resilience', 'Connectivity', 'Island Size', 'Largest Void', 'Void Size Change', 'Coarsening', 'Intensity Difference Area 1', 'Intensity Difference Area 2', 'Average Velocity', 'Average Speed', 'Average Divergence', 'Island Movement Direction', 'Flow Direction']
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

def create_barcode(figpath, entry, bar_gen=True):
    # Define color mappings
    binary_colors = {0: [0, 0, 0], 1: [1, 1, 1], None: [0.5, 0.5, 0.5]}  # Black for 0, white for 1
    colormap = plt.get_cmap('plasma')  # Colormap for floats

    # channel, r, c, spanning, void_value, island_size, island_movement, void_growth, direct, avg_vel, avg_speed, avg_div, c_area1, c_area2 = entry
    channel = entry[0]
    binary_indices = [1, 2, 6]
    binary_values = [entry[val] for val in binary_indices]
    float_indices = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13]
    float_values = [entry[val] for val in float_indices]

    # Define normalization limits of floating point values
    bin_size_lim = [0, 1]  # Size limit for void and island size
    direct_lim = [-np.pi, np.pi] # Limits on the direction of the island and flow (radians)
    void_growth_lim = [0, 5] # Limits for the expected growth of the void
    avg_vel_lim = [0, 10] # Limit for the average velocity (pixels/sec)
    avg_speed_lim = [0, 10] # Limit for the average speed (pixels/sec)
    avg_div = [-1, 1] # Limit for divergence metric (-1 is pure contraction, 1 is pure expansion)
    i_area_lim = [0, 0.2] # Limit for the first two areas of the delta-I distribution
    limits = [bin_size_lim, bin_size_lim, void_growth_lim, i_area_lim, i_area_lim, avg_vel_lim, avg_speed_lim, avg_div, direct_lim, direct_lim]
    
    def normalize(x, min_float, max_float):
        if x == None:
            return None
        return (x - min_float) / (max_float - min_float)
    
    # Create the color barcode
    barcode = [None] * 13
    for index, value in zip(binary_indices, binary_values):
        color = binary_colors[value]
        barcode[index - 1] = color
    for f_index, entry_value_float, float_lim in zip(float_indices, float_values, limits):
        cval = normalize(entry_value_float, float_lim[0], float_lim[1])
        color = [0.5, 0.5, 0.5] if cval == None else colormap(cval)[:3]
        barcode[f_index - 1] = color
    
    # Convert to numpy array and reshape for plotting
    barcode = np.array(barcode)
    if bar_gen == True:
        barcode_image = np.tile(barcode, (10, 1, 1))  # Repeat the barcode to make it visible
        # Plot and save the barcode
        plt.imshow(barcode_image, aspect='auto')
        plt.axis('off')
        plt.savefig(figpath, bbox_inches='tight', pad_inches=0)

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