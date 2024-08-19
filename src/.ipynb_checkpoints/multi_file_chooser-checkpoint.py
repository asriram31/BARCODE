from gooey import Gooey, GooeyParser
import numpy as np
import csv

@Gooey
def main():
    parser = GooeyParser()
    parser.add_argument('--csv_paths', metavar = 'CSV File Locations', widget='MultiFileChooser', help="Select the CSV files representing the datasets you would like to combine", gooey_options = {
        'wildcard': "CSV Document (*.csv)|*.csv"})
    parser.add_argument('--combined_location', metavar = 'Aggregate Location', widget='FileSaver', help="Select a location for the aggregate CSV file to be located", gooey_options = {
        'default_file': "aggregate_summary.csv"
    })
    parser.add_argument('--generate_agg_barcode', metavar = 'Generate Aggregate Barcode', widget='CheckBox', help="Click to generate an aggregate barcode from these files", action="store_true")
    parser.add_argument('--normalize_agg_barcode', metavar = 'Normalize Aggregate Barcode', widget='CheckBox', help="Click to normalize the barcode (color will be determined by the limits of the dataset)", action='store_true')


    settings = parser.parse_args()

    files = settings.csv_paths.split(',')
    combined_csv_loc = settings.combined_location

    gen_barcode = settings.generate_agg_barcode
    normalize_data = settings.normalize_agg_barcode
    
    if gen_barcode:
        combined_barcode_loc = os.path.join(os.path.dirname(combined_csv_loc), 'aggregate_barcode.png')

    headers = ['Channel', 'Resilience', 'Connectivity', 'Island Size', 'Largest Void', 'Void Size Change', 'Coarsening', 'Intensity Difference Area 1', 'Intensity Difference Area 2', 'Average Velocity', 'Average Speed', 'Average Divergence', 'Island Movement Direction', 'Flow Direction']
    f = open(combined_csvLoc, 'w') # Clears the CSV file if it already exists, and creates it if it does not
    csv_writer = csv.writer(f)
    csv_writer.writerow(headers)
    f.close()

    def combine_csvs(csv_list):
        filenames = []
        csv_data = np.zeros(shape=(14))
        if not csv_list:
            return None
        for csv_file in csv_list:
            with open(csv_file, 'r') as fread, open(combined_csv_loc, 'a') as fwrite:
                csv_reader = csv.reader(fread)
                csv_writer = csv.writer(fwrite)
                next(csv_reader, None)
                for row in csv_reader:
                    if len(row) == 1:
                        filenames.append(str(row))
                    else:
                        arr_row = np.array(row)
                        csv_data = np.vstack(csv_data, arr_row)
                    csv_writer.writerow(row)

    if gen_barcode:
        csv_data_no_channels = csv_data[:,1:]
        gen_combined_barcode(csv_data_no_channels, normalize_data)

    # print(settings.file_path)

def gen_combined_barcode(data, normalize_data = True):
    # Order: [channel, r, spanning, island_size, void_value, void_growth,  c, c_area1, c_area2, avg_vel, avg_speed, avg_div, island_movement, direct]
    resilience = data[:,0]
    connectivity = data[:,1]
    island_size = data[:,2]
    void_value = data[:,3]
    void_growth = data[:,4]
    coarsening = data[:,5]
    i_diff_a1 = data[:,6]
    i_diff_a2 = data[:,7]
    avg_vel = data[:,8]
    avg_speed = data[:,9]
    avg_div = data[:,10]
    island_dir = data[:,11]
    flow_dir = data[:,12]
    all_entries = [resilience, connectivity, island_size, void_value, void_growth, coarsening, i_diff_a1, i_diff_a2, avg_vel, avg_speed, avg_div, island_dir, flow_dir]

    # Define normalization limits of floating point values
    bin_size_lim = [0, 1]  # Size limit for void and island size
    direct_lim = [-np.pi, np.pi] # Limits on the direction of the island and flow (radians)
    void_growth_lim = [0, 5] # Limits for the expected growth of the void
    avg_vel_lim = [0, 10] # Limit for the average velocity (pixels/sec)
    avg_speed_lim = [0, 10] # Limit for the average speed (pixels/sec)
    avg_div = [-1, 1] # Limit for divergence metric (-1 is pure contraction, 1 is pure expansion)
    i_area_lim = [0, 0.2] # Limit for the first two areas of the delta-I distribution
    limits = [bin_size_lim, bin_size_lim, void_growth_lim, i_area_lim, i_area_lim, avg_vel_lim, avg_speed_lim, avg_div, direct_lim, direct_lim]

    if normalize_data:
        for i, lim in enumerate(limits):
            limits[i] = [np.min(all_entries[i]), np.max(all_entries[i])]

    binary_colors = {0: [0, 0, 0], 1: [1, 1, 1], None: [0.5, 0.5, 0.5]}  # Black for 0, white for 1
    colormap = plt.get_cmap('plasma')  # Colormap for floats

    def normalize(x, min_float, max_float):
        if x == None:
            return None
        return (x - min_float) / (max_float - min_float)

    binary_indices = [0, 1, 5]
    binary_values = [all_entries[idx] for val in binary_indices]
    float_indices = [2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
    float_values = [all_entries[val] for val in float_indices]

    agg_barcode = np.zeros(13)
    for row in range(len(data)):
        barcode = [None] * 13
        for idx in range(len(all_entries)):
            if idx in binary_indices:
                value = data[row, ]
            elif idx in float_indices:
                
        
    
    # Create the color barcode
    
    for index, value in zip(binary_indices, binary_values):
        color = binary_colors[value]
        barcode[index - 1] = color
    for f_index, entry_value_float, float_lim in zip(float_indices, float_values, limits):
        cval = normalize(entry_value_float, float_lim[0], float_lim[1])
        color = [0.5, 0.5, 0.5] if cval == None else colormap(cval)[:3]
        barcode[f_index - 1] = color
    

if __name__ == "__main__":
    main()