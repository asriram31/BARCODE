from reader import read_file
import os, csv, sys, yaml, time
from resilience_tracker import check_resilience
from flow_tracker import check_flow
from coarse_tracker import check_coarse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec
from gooey import Gooey, GooeyParser

def execute_htp(filepath, config_data):
    reader_data = config_data['reader']
    channel_select, resilience, flow, coarsening, verbose, accept_dim_im, accept_dim_channel = reader_data.values()
    r_data = config_data['resilience_parameters']
    f_data = config_data['flow_parameters']
    c_data = config_data['coarse_parameters']

    
    def check(channel, resilience, flow, coarse, resilience_data, flow_data, coarse_data):
        figure_dir_name = remove_extension(filepath) + ' Plots'
        fig_channel_dir_name = os.path.join(figure_dir_name, 'Channel ' + str(channel))
        if not os.path.exists(figure_dir_name):
            os.makedirs(figure_dir_name)
        if not os.path.exists(fig_channel_dir_name):
            os.makedirs(fig_channel_dir_name)
        
        if resilience == True:
            r_offset = resilience_data['r_offset']
            pt_loss, pt_gain = resilience_data['percent_threshold'].values()
            f_step = resilience_data['frame_step']
            f_start, f_stop = resilience_data['evaluation_settings'].values()
            r, rfig, void_value, spanning, island_size, island_movement, void_growth = check_resilience(file, channel, r_offset, pt_loss, pt_gain, f_step, f_start, f_stop)
        else:
            r = None
            rfig = None
            spanning = None
            void_value = None
            island_size = None
            island_movement = None
            void_growth = None
        if flow == True:
            frame_step, downsample = flow_data.values()
            direct, avg_vel, avg_speed, avg_div = check_flow(file, fig_channel_dir_name, channel, int(frame_step), downsample)
        else:
            direct = None
            avg_vel = None
            avg_speed = None
            avg_div = None
        if coarse == True:
            fframe, lframe = coarse_data['evaluation_settings'].values()
            t_percent = coarse_data['threshold_percentage']
            percent_frames = coarse_data['mean_mode_frames_percent']
            c, cfig, c_area1, c_area2 = check_coarse(file, channel, fframe, lframe, t_percent, percent_frames)
        else:
            c = None
            cfig = None
            c_area1 = None
            c_area2 = None

        figpath = os.path.join(fig_channel_dir_name, 'Summary Graphs.png')
        if verbose == True:
            fig = plt.figure(figsize = (15, 5))
            gs = gridspec.GridSpec(1,3)

            if rfig != None:
                ax1 = rfig.axes[0]
                ax1.remove()
                ax1.figure = fig
                fig.add_axes(ax1)
                ax1.set_position([2.5/15, 1/10, 4/5, 4/5])

            if cfig != None:               
                ax3 = cfig.axes[0]
                ax3.remove()
                ax3.figure = fig
                fig.add_axes(ax3)
                ax3.set_position([32.5/15, 1/10, 4/5, 4/5])

            plt.savefig(figpath)
        plt.close(rfig)
        plt.close(cfig)
        plt.close(fig)

        result = [channel, r, spanning, island_size, void_value, void_growth,  c, c_area1, c_area2, avg_vel, avg_speed, avg_div, island_movement, direct]

        figpath2 = os.path.join(fig_channel_dir_name, 'Channel ' + str(channel) + 'Barcode.png')
        create_barcode(figpath2, result)
            
        return result
    
    file = read_file(filepath, accept_dim_im)


    if (isinstance(file, np.ndarray) == False):
        return None

    channels = min(file.shape)
    
    if (isinstance(channel_select, int) == False and isinstance(channel_select, bool) == False) or channel_select > channels:
        raise ValueError("Please give correct channel input (False for all channels, -1 for the last channel, 0 for channel 1, etc)")
    
    rfc = []
    if channel_select == 'All':
        print('Total Channels:', channels)
        channel = channels - 1
        print(file[:,:,:,channel])
        # for channel in range(channels):
        print('Channel:', channel)
        #     if check_channel_dim(file[:,:,:,channel]) and not accept_dim_channel:
        #         print('Channel too dim, not enough signal, skipping...')
        #         continue
        results = check(channel, resilience, flow, coarsening, r_data, f_data, c_data)
        rfc.append(results)
    
    else:
        print('Channel: ', channel_select)
        results = check(channel_select, resilience, flow, coarsening, r_data, f_data, c_data)
        rfc.append(results)

    return rfc

def remove_extension(filepath):
    if filepath.endswith('.tiff'):
        return filepath.removesuffix('.tiff')
    if filepath.endswith('.tif'):
        return filepath.removesuffix('.tif')
    if filepath.endswith('.nd2'):
        return filepath.removesuffix('.nd2')

def writer(output_filepath, data):
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

def process_directory(root_dir, config_data):
    
    if os.path.isfile(root_dir):
        all_data = []
        file_path = root_dir
        filename = os.path.basename(file_path)
        dir_name = os.path.dirname(file_path)
        start_time = time.time()
        rfc_data = execute_htp(file_path, config_data)
        if rfc_data == None:
            raise TypeError("Please input valid file type ('.nd2', '.tiff', '.tif')")
        all_data.append([filename])
        all_data.extend(rfc_data)
        all_data.append([])

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Time Elapsed:', elapsed_time)

        output_filepath = os.path.join(dir_name, filename + 'summary.csv')

        writer(output_filepath, all_data)
    else: 
        all_data = []
        start_folder_time = time.time()
        for dirpath, dirnames, filenames in os.walk(root_dir):
    
            dirnames[:] = [d for d in dirnames if d not in ["To Be Tested", "Aditya", "htp-screening"]]
    
            for filename in filenames:
                if filename.startswith('._'):
                    continue
                file_path = os.path.join(dirpath, filename)
                start_time = time.time()
                try:
                    rfc_data = execute_htp(file_path, config_data)
                except Exception as e:
                    with open(os.path.join(root_dir, "failed_files_txt"), "a") as log_file:
                        log_file.write(f"FileL {file_path}, Exception: {str(e)}\n")
                    continue
                if rfc_data == None:
                    continue
                all_data.append([file_path])
                all_data.extend(rfc_data)
                all_data.append([])

                end_time = time.time()
                elapsed_time = end_time - start_time
                print('Time Elapsed:', elapsed_time)
        
        output_filepath = os.path.join(root_dir, "summary.csv")
        
        writer(output_filepath, all_data)
        end_folder_time = time.time()
        elapsed_folder_time = end_folder_time - start_folder_time
        print('Time Elapsed to Process Folder:', elapsed_folder_time)

def create_barcode(figpath, entry):
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
    bin_size_lim = [0, 1]
    direct_lim = [-np.pi, np.pi]
    void_growth_lim = [0, 5]
    avg_vel_lim = [0, 10]
    avg_speed_lim = [0, 10]
    avg_div = [-1, 1]
    i_area_lim = [0, 1]
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
    barcode_image = np.tile(barcode, (10, 1, 1))  # Repeat the barcode to make it visible

    # Plot and save the barcode
    plt.imshow(barcode_image, aspect='auto')
    plt.axis('off')
    plt.savefig(figpath, bbox_inches='tight', pad_inches=0)

def check_channel_dim(image):
    min_intensity = np.min(image)
    mean_intensity = np.mean(image)
    return 2 * np.exp(-1) * mean_intensity <= min_intensity

@Gooey(program_name="DMREF BARCODE Program", tabbed_groups=True, navigation='Tabbed')
def main():
    parser = GooeyParser(description='Code that runs through the BARCODE code developed by the DMREF group')

    gc = parser.add_argument_group("Execution Settings")
    fdc = gc.add_mutually_exclusive_group()
    fdc.add_argument('--file_path', metavar = 'File Chooser', widget='FileChooser', gooey_options = {
        'wildcard': "Document (*.nd2)|*.nd2|"
        "TIFF Image (*.tiff)|*.tiff|"
        "TIFF Image (*.tif)|*.tif"
    }
    )
    fdc.add_argument('--dir_path', metavar='Directory Chooser', widget='DirChooser')
    
    c_select = gc.add_mutually_exclusive_group()
    c_select.add_argument('--channels', metavar='Parse All Channels', widget='CheckBox', action='store_true')

    c_select.add_argument('--channel_selection', metavar='Choose Channel', widget='IntegerField', gooey_options = {
        'min': -3, 
        'max': 4
    }
    )

    gc.add_argument('--check_resilience', metavar='Resilience', help='Evaluate sample(s) using binarization module', widget='CheckBox', action='store_true')
    gc.add_argument('--check_flow', metavar='Flow', help='Evaluate sample(s) using optical flow module', widget='CheckBox', action='store_true')
    gc.add_argument('--check_coarsening', metavar='Coarsening', help='Evaluate sample(s) using intensity distribution module', widget='CheckBox', action='store_true')
    
    gc.add_argument('--verbose', metavar='Verbose', help='Show more details', widget='CheckBox', action='store_true')
    gc.add_argument('--return_graphs', metavar='Save Graphs', help='Click to save graphs representing sample changes', widget='CheckBox', action='store_true')
    gc.add_argument('--return_intermediates', metavar='Intermediates', help='Click to save intermediate data structures (flow fields, binarized images, intensity distributions)', widget='CheckBox', action='store_true')

    gc.add_argument('--dim_images', metavar='Dim Images', help='Click to scan files that may be too dim to accurately profile', widget='CheckBox', action='store_true')
    gc.add_argument('--dim_channels', metavar='Dim Channels', help='Click to scan channels that may be too dim to accurately profile', widget='CheckBox', action='store_true')

    res_settings = parser.add_argument_group('Resilience Settings')
    res_settings.add_argument('--r_offset', metavar='Binarization Threshold', help='Adjust the pixel intensity threshold as a percentage of the mean (0 - 200%)', widget='DecimalField', gooey_options = {
        'min':-1.0,
        'max':1.0,
        'increment':0.05
    })
    res_settings.add_argument('--pt_loss', metavar='Percent Threshold Loss', help="Percentage of original void size that final void size must be greater than to considered resilient", widget='DecimalField', gooey_options = {
        'min':0,
        'max':1,
        'increment':0.05 
    })
    res_settings.add_argument('--pt_gain', metavar='Percent Threshold Gain', help="Percentage of original void size that final void size must be less than to considered resilient", widget='DecimalField', gooey_options = {
        'min':1,
        'max':5,
        'increment':0.05 
    })

    res_settings.add_argument('--res_f_step', metavar = 'Frame Step', help = "Controls how many frames between evaluated frames", widget='Slider', gooey_options = {
        'min':1,
        'increment':1
    })

    res_settings.add_argument('--pf_start', metavar='Frame Start Percent', help="Determines starting percentage of frames to evaluate for resilience", widget='DecimalField', gooey_options = {
        'min':0.5,
        'max':0.9,
        'increment':0.05
    })

    res_settings.add_argument('--pf_stop', metavar='Frame Stop Percent', help="Determines ending percentage of frames to evaluate for resilience", widget='DecimalField', gooey_options = {
        'min':0.9,
        'max':1,
        'increment':0.05
    })

    flow_settings = parser.add_argument_group('Flow Settings')

    flow_settings.add_argument('--flow_f_step', metavar = 'Frame Step', help = "Controls the interval between frames the flow field is calculated at", widget = 'Slider', gooey_options = {
        'min':1,
        'increment':1
    })

    flow_settings.add_argument('--downsample', metavar = 'Downsample', help = "Controls the downsampling rate of the flow field (larger values give less precision, less prone to noise)", widget = 'IntegerField', gooey_options = {
        'min':1,
        'increment':1
    })

    coarse_settings = parser.add_argument_group('Coarsening Settings')

    coarse_settings.add_argument('--first_frame', metavar='First Frame', help = 'Controls which frame is used as the first frame for intensity distribution comparisons', widget='Slider', gooey_options = {
        'min':1,
        'increment':1
    })

    final_frame = coarse_settings.add_mutually_exclusive_group()

    final_frame.add_argument('--eval_last_frame', metavar = 'Use Default Last Frame', help = 'Use final frame of image for intensity distribution comparisons', action='store_true')

    final_frame.add_argument('--select_last_frame', metavar = 'Select Last Frame', help = "Select the final frame of the video for intensity distribution comparisons", widget = 'IntegerField')

    coarse_settings.add_argument('--thresh_percent', metavar = 'Threshold Percentage', help = 'Select the threshold percentage mean-mode difference between the final and initial intensity distributions; adjust for different objective lenses (5-7 for 60x objective lens, 1 for 20x, etc)', widget = 'Slider', gooey_options = {
        'min': 1,
        'max': 8,
        'increment': 1
    })

    coarse_settings.add_argument('--pf_evaluation', metavar = 'Percent of Frames Evaluated', help = "Determine what percent of frames are evaluated for intensity distribution comparison using mean-mode comparison", widget = 'DecimalField', gooey_options = {
        'min':0.01,
        'max': 0.2,
        'increment':0.01
    })
    
    settings = parser.parse_args()
    
    # abs_path = os.path.abspath(sys.argv[0])

    dir_name = settings.dir_path if settings.dir_path != None else settings.file_path

    config_data = set_config_data(settings)

    # dir_name = sys.argv[1]
    
    process_directory(dir_name, config_data)

def set_config_data(args = None):
    config_data = {}
    reader_data = {}
    writer_data = {}
    resilience_data = {}
    flow_data = {}
    coarsening_data = {}
    if args:
        reader_data = {
            'channel_select':'All' if args.channels else int(args.channel_selection),
            'resilience':args.check_resilience,
            'flow':args.check_flow,
            'coarsening':args.check_coarsening,
            'verbose':args.verbose,
            'accept_dim_images':args.dim_images,
            'accept_dim_channels':args.dim_channels
        }
        if reader_data['resilience']:
            resilience_data = {
                'r_offset':float(args.r_offset),
                'percent_threshold':{'pt_loss':float(args.pt_loss), 'pt_gain':float(args.pt_gain)},
                'frame_step':int(args.res_f_step),
                'evaluation_settings':{'f_start':float(args.pf_start), 'f_stop':float(args.pf_stop)},
            }
        if reader_data['flow']:
            flow_data = {
                'frame_step':int(args.flow_f_step),
                'downsample':int(args.downsample)
            }

        if reader_data['coarsening']:
            coarsening_data = {
                'evaluation_settings':{'first_frame':int(args.first_frame), 'last_frame':False if args.eval_last_frame else args.select_last_frame},
                'threshold_percentage':float(args.thresh_percent),
                'mean_mode_frames_percent':float(args.pf_evaluation),
            }

        config_data = {
            'reader':reader_data,
            'resilience_parameters':resilience_data,
            'flow_parameters':flow_data,
            'coarse_parameters':coarsening_data
        }
        
    return config_data

if __name__ == "__main__":
    main()
