from reader import read_file
import os, csv, sys, yaml, time
from resilience_tracker import check_resilience
from flow_tracker import check_flow
from coarse_tracker import check_coarse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import gridspec


def execute_htp(filepath, config_data):
    reader_data = config_data['reader']
    channel_select, resilience, flow, coarsening, verbose, accept_dim = reader_data.values()
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
        if verbose == True:
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
            
        return [channel, r, c, void_value, spanning, island_size, island_movement, void_growth, direct, avg_vel, avg_speed, avg_div, c_area1, c_area2]
    
    file = read_file(filepath, accept_dim)

    if (isinstance(file, np.ndarray) == False):
        return None

    channels = min(file.shape)
    
    if (isinstance(channel_select, int) == False) or channel_select > channels:
        raise ValueError("Please give correct channel input (-1 for all channels, 0 for channel 1, etc)")
    
    rfc = []
    
    if channel_select == -1:
        print('Total Channels:', channels)
        for channel in range(channels):
            print('Channel:', channel)
            rfc.append(check(channel, resilience, flow, coarsening, r_data, f_data, c_data))
    
    else:
        print('Channel: ', channel_select)
        rfc.append(check(channel_select, resilience, flow, coarsening, r_data, f_data, c_data))

    return rfc

def remove_extension(filepath):
    if filepath.endswith('.tiff'):
        return filepath.removesuffix('.tiff')
    if filepath.endswith('.tif'):
        return filepath.removesuffix('.tif')
    if filepath.endswith('.nd2'):
        return filepath.removesuffix('.nd2')

def writer(data, directory):
    if data:
        headers = ['Channel', 'Resilience', 'Coarseness', 'Largest void', 'Span', 'Island Size', 'Island Movement', 'Void Growth', "Flow Direction", "Average Velocity", "Average Speed", "Average Divergence", 'Intensity Difference Area 1', 'Intensity Difference Area 2']
        output_filepath = os.path.join(directory, "summary.csv")
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

        writer(all_data, dir_name)
    else: 
        all_data = []
        start_folder_time = time.time()
        for dirpath, dirnames, filenames in os.walk(root_dir):
    
            dirnames[:] = [d for d in dirnames]
    
            for filename in filenames:
                if filename.startswith('._'):
                    continue
                file_path = os.path.join(dirpath, filename)
                print(file_path)
                start_time = time.time()
                rfc_data = execute_htp(file_path, config_data)
                if rfc_data == None:
                    continue
                all_data.append([file_path])
                all_data.extend(rfc_data)
                all_data.append([])

                end_time = time.time()
                elapsed_time = end_time - start_time
                print('Time Elapsed:', elapsed_time)

        writer(all_data, root_dir)
        end_folder_time = time.time()
        elapsed_folder_time = end_folder_time - start_folder_time
        print('Time Elapsed to Process Folder:', elapsed_folder_time)

        

def main():
    abs_path = os.path.abspath(sys.argv[0])
    dir_name = sys.argv[1]
    if len(sys.argv) == 3:
        config_path = sys.argv[2]
    else:
        # Update this with your filepath -- if your directory is htp-screening-main, use that as the highest level directory instead
        config_path = os.path.join(os.path.dirname(abs_path), 'config.yaml')
    with open(config_path, "r") as yamlfile:
        config_data = yaml.load(yamlfile, Loader=yaml.CLoader)
        process_directory(dir_name, config_data)

if __name__ == "__main__":
    main()
