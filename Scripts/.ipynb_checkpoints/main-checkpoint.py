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
from writer import write_file, create_barcode

def execute_htp(filepath, config_data):
    reader_data = config_data['reader']
    channel_select, resilience, flow, coarsening, verbose, return_graphs, accept_dim_im, accept_dim_channel, excluded_dirs = reader_data.values()
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
            r = "Resilience not tested"
            rfig = None
            spanning = None
            void_value = None
            island_size = None
            island_movement = None
            void_growth = None
        if flow == True:
            mcorr_len, min_fraction, frame_step, downsample, pix_size, bin_width = flow_data.values()
            ffig, direct, avg_vel, avg_speed, avg_div = check_flow(file, fig_channel_dir_name, channel, mcorr_len, min_fraction, frame_step, downsample, pix_size, bin_width)
        else:
            ffig = None
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
            c = "Coarseness not tested."
            cfig = None
            c_area1 = None
            c_area2 = None

        figpath = os.path.join(fig_channel_dir_name, 'Summary Graphs.png')
        if return_graphs == True:
            fig = plt.figure(figsize = (15, 5))
            gs = gridspec.GridSpec(1,3)

            if rfig != None:
                ax1 = rfig.axes[0]
                ax1.remove()
                ax1.figure = fig
                fig.add_axes(ax1)
                ax1.set_position([2.5/15, 1/10, 4/5, 4/5])

            if ffig != None:

                ax2 = ffig.axes[0]
                ax2.remove()
                ax2.figure = fig
                fig.add_axes(ax2)
                ax2.set_position([17.5/15, 1/10, 4/5, 4/5])

            if cfig != None:               
                ax3 = cfig.axes[0]
                ax3.remove()
                ax3.figure = fig
                fig.add_axes(ax3)
                ax3.set_position([32.5/15, 1/10, 4/5, 4/5])

            plt.savefig(figpath)

        plt.close(rfig)
        plt.close(ffig)
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
    
    if (isinstance(channel_select, int) == False and channel_select != False) or channel_select > channels or channel_select == None or (channels + channel_select < 0):
        raise ValueError("Please give correct channel input (False for all channels, 0 for channel 1, -1 for the last channel, etc)")
    
    rfc = []
    
    if channel_select == False:
        if verbose:
            print('Total Channels:', channels)
        for channel in range(channels):
            if verbose:
                print('Channel:', channel)
            if check_channel_dim(file[:,:,:,channel]) and not accept_dim_channel and verbose:
                print('Channel too dim, not enough signal, skipping...')
                continue
        results = check(channel, resilience, flow, coarsening, r_data, f_data, c_data)
        rfc.append(results)
    
    else:
        if channel_select < 0:
            channel_select = channels + channel_select
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


def process_directory(root_dir, config_data, normalize_dataset):
    excluded_dirs = config_data['reader']['exclude_directories'].values()
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

        write_file(output_filepath, all_data)
    else: 
        all_data = []
        start_folder_time = time.time()
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
            for filename in filenames:
                if filename.startswith('._'):
                    continue
                file_path = os.path.join(dirpath, filename)
                start_time = time.time()
                try:
                    rfc_data = execute_htp(file_path, config_data)
                except Exception as e:
                    with open(os.path.join(root_dir, "failed_files.txt"), "a") as log_file:
                        log_file.write(f"File {file_path}, Exception: {str(e)}\n")
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
        
        write_file(output_filepath, all_data)
        end_folder_time = time.time()
        elapsed_folder_time = end_folder_time - start_folder_time
        print('Time Elapsed to Process Folder:', elapsed_folder_time)

def check_channel_dim(image):
    min_intensity = np.min(image)
    mean_intensity = np.mean(image)
    return 2 * np.exp(-1) * mean_intensity <= min_intensity


def main():
    abs_path = os.path.abspath(sys.argv[0])
    dir_name = sys.argv[1]
    if len(sys.argv) == 3:
        config_path = sys.argv[2]
    else:
        config_path = os.path.join(os.path.dirname(abs_path), 'config.yaml')
    with open(config_path, "r") as yamlfile:
        config_data = yaml.load(yamlfile, Loader=yaml.CLoader)
        writer_data = config_data['writer']
        normalize_dataset, generate_rgbmap, generate_barcode = writer_data.values()
        process_directory(dir_name, config_data, normalize_dataset)

if __name__ == "__main__":
    main()
