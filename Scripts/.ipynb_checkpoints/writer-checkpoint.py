import csv
import matplotlib.pyplot as plt
import numpy as np

def write_file(output_filepath, data):
    if data:
        headers = ['Channel', 'Resilience', 'Connectivity', 'Island Size', 'Largest Void', 'Void Size Change', 'Coarsening', 'Intensity Difference Area 1', 'Intesity Difference Area 2', 'Average Velocity', 'Average Speed', 'Average Divergence', 'Island Movement Direction', 'Flow Direction']
        with open(output_filepath, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for entry in data:
                if isinstance(entry, list) and len(entry) == 1:
                    # Write the file name
                    csvwriter.writerow(entry)
                    csvwriter.writerow(headers)  # Write headers after the filename
                elif entry:
                    # Write the headers if entry contains channel data
                    csvwriter.writerow(headers)
                    headers = []  # Ensure headers are only written once per file
                    csvwriter.writerow(entry)
                else:
                    # Write an empty row
                    csvwriter.writerow([])
def create_barcode(figpath, entry, use_dataset_limits=False):
    # Define color mappings
    binary_colors = {0: [0, 0, 0], 1: [1, 1, 1]}  # Black for 0, white for 1
    colormap = plt.get_cmap('plasma')  # Colormap for floats

    # channel, r, c, spanning, void_value, island_size, island_movement, void_growth, direct, avg_vel, avg_speed, avg_div, c_area1, c_area2 = entry
    channel = entry[0]
    binary_indices = [1, 2, 6]
    binary_values = [entry[val] for val in binary_indices]
    float_indices = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13]
    float_values = [entry[val] for val in float_indices]
    # Define normalization limits of floating point values
    if use_dataset_limits == False:
        bin_size_lim = [0, 1]
        direct_lim = [-np.pi, np.pi]
        void_growth_lim = [0, 5]
        avg_vel_lim = [0, 10]
        avg_speed_lim = [0, 10]
        avg_div = [-1, 1]
        i_area_lim = [0, 1]
        limits = [bin_size_lim, bin_size_lim, void_growth_lim, i_area_lim, i_area_lim, avg_vel_lim, avg_speed_lim, avg_div, direct_lim, direct_lim]
    
    def normalize(x, min_float, max_float):
        return (x - min_float) / (max_float - min_float)
    
    # Create the color barcode
    barcode = [None] * 13
    for index, value in zip(binary_indices, binary_values):
        color = binary_colors[value]
        barcode[index - 1] = color
    for f_index, entry_value_float, float_lim in zip(float_indices, float_values, limits):
        color = colormap(normalize(entry_value_float, float_lim[0], float_lim[1]))[:3]
        barcode[f_index - 1] = color
    
    # Convert to numpy array and reshape for plotting
    barcode = np.array(barcode)
    barcode_image = np.tile(barcode, (10, 1, 1))  # Repeat the barcode to make it visible

    # Plot and save the barcode
    plt.imshow(barcode_image, aspect='auto')
    plt.axis('off')
    plt.savefig(figpath, bbox_inches='tight', pad_inches=0)
