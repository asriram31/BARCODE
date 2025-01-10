from reader import read_file
import os, yaml, time, functools, builtins
from binarization import check_resilience
from flow import check_flow
from intensity_distribution_comparison import check_coarse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from writer import write_file, gen_combined_barcode

class MyException(Exception):
    pass

def check_channel_dim(image):
    min_intensity = np.min(image)
    mean_intensity = np.mean(image)
    return 2 * np.exp(-1) * mean_intensity <= min_intensity

def execute_htp(filepath, config_data, fail_file_loc, count, total):
    reader_data = config_data['reader']
    save_intermediates = config_data['writer']['return_intermediates']
    accept_dim_channel, accept_dim_im, channel_select, coarsening, flow, resilience, return_graphs, verbose = reader_data.values()
    r_data = config_data['resilience_parameters']
    f_data = config_data['flow_parameters']
    c_data = config_data['coarse_parameters']
    
    print = functools.partial(builtins.print, flush=True)
    vprint = print if verbose else lambda *a, **k: None

    def check(file_path, channel, resilience, flow, coarse, resilience_data, flow_data, coarse_data, fail_file_loc):
        flag = None
        figure_dir_name = remove_extension(filepath) + ' BARCODE Output'
        fig_channel_dir_name = os.path.join(figure_dir_name, 'Channel ' + str(channel))
        if not os.path.exists(figure_dir_name):
            os.makedirs(figure_dir_name)
        if not os.path.exists(fig_channel_dir_name):
            os.makedirs(fig_channel_dir_name)
        
        if resilience == True:
            r_offset = resilience_data['r_offset']
            f_step = resilience_data['frame_step']
            f_start = resilience_data['evaluation_settings']['f_start']
            f_stop = resilience_data['evaluation_settings']['f_stop']
            try:
                rfig, binarization_outputs = check_resilience(file, fig_channel_dir_name, channel, r_offset, f_step, f_start, f_stop, return_graphs, save_intermediates, verbose)
            except Exception as e:
                with open(fail_file_loc, "a", encoding="utf-8") as log_file:
                    log_file.write(f"File: {file_path}, Module: Binarization, Exception: {str(e)}\n")
                rfig = None
                binarization_outputs = [None] * 7
        else:
            rfig = None
            binarization_outputs = [None] * 7
        if flow == True:
            downsample = flow_data['downsample']
            frame_interval = flow_data['frame_interval']
            frame_step = flow_data['frame_step']
            nm_pix_ratio = flow_data['nm_pixel_ratio']
            win_size = flow_data['win_size']
            try:
                flow_outputs = check_flow(file, fig_channel_dir_name, channel, int(frame_step), downsample, frame_interval, nm_pix_ratio, return_graphs, save_intermediates, verbose, int(win_size))
            except Exception as e:
                with open(fail_file_loc, "a", encoding="utf-8") as log_file:
                    log_file.write(f"File: {file_path}, Module: Optical Flow, Exception: {str(e)}\n")
                flow_outputs = [None] * 4
        else:
            flow_outputs = [None] * 4
        if coarse == True:
            fframe = coarse_data['evaluation_settings']['first_frame']
            lframe = coarse_data['evaluation_settings']['last_frame']
            percent_frames = coarse_data['mean_mode_frames_percent']
            try:
                cfig, id_outputs, flag = check_coarse(file, fig_channel_dir_name, channel, fframe, lframe, percent_frames, save_intermediates, verbose)
            except Exception as e:
                with open(fail_file_loc, "a", encoding="utf-8") as log_file:
                    log_file.write(f"File: {file_path}, Module: Intensity Distribution, Exception: {str(e)}\n")
                cfig = None
                id_outputs = [None] * 6
                flag = None
        else:
            cfig = None
            id_outputs = [None] * 6
            flag = None

        figpath = os.path.join(fig_channel_dir_name, 'Summary Graphs.png')
        if return_graphs == True:
            fig = plt.figure(figsize = (10, 5))
            gs = gridspec.GridSpec(1,2)

            if rfig != None:
                ax1 = rfig.axes[0]
                ax1.remove()
                ax1.figure = fig
                fig.add_axes(ax1)
                ax1.set_position([1.5/10, 1/10, 4/5, 4/5])

            if cfig != None:               
                ax3 = cfig.axes[0]
                ax3.remove()
                ax3.figure = fig
                fig.add_axes(ax3)
                ax3.set_position([11.5/10, 1/10, 4/5, 4/5])

            plt.savefig(figpath)
            plt.close(fig)
        plt.close('all')

        result = [channel] + [flag] + binarization_outputs + id_outputs + flow_outputs        
        vprint('Channel Screening Completed')
            
        return result
    
    file = read_file(filepath, count, total, accept_dim_im)
    if (isinstance(file, np.ndarray) == False):
        raise TypeError("File was not of the correct filetype")
    
    channels = min(file.shape)
    
    rfc = []
    if channel_select == 'All':
        vprint('Total Channels:', channels)
        for channel in range(channels):
            vprint('Channel:', channel)
            if check_channel_dim(file[:,:,:,channel]) and not accept_dim_channel:
                vprint('Channel too dim, not enough signal, skipping...')
                continue
            elif check_channel_dim(file[:,:,:,channel]) and accept_dim_channel:
                vprint('Warning: channel is dim. Accuracy of screening may be limited by this.')
                results = check(filepath, channel, resilience, flow, coarsening, r_data, f_data, c_data, fail_file_loc)
                results[1] = results[1] + 1
                
            else:
                results = check(filepath, channel, resilience, flow, coarsening, r_data, f_data, c_data, fail_file_loc)
            rfc.append(results)
    
    else:
        while channel_select < 0:
            channel_select = channels + channel_select # -1 will correspond to last channel, etc
        if channel_select >= channels:
            channel_select = channels - 1 # Sets channel to maximum channel if channel selected is out of range
        vprint('Channel: ', channel_select)
        if check_channel_dim(file[:,:,:,channel_select]):
            vprint('Warning: channel is dim. Accuracy of screening may be limited by this.')
            results = check(filepath, channel_select, resilience, flow, coarsening, r_data, f_data, c_data, fail_file_loc)
            results[1] = results[1] + 1 # Indicate dim channel flag present
        else:
            results = check(filepath, channel_select, resilience, flow, coarsening, r_data, f_data, c_data, fail_file_loc)
        rfc.append(results)

    return rfc, count + 1

def remove_extension(filepath):
    if filepath.endswith('.tif'):
        return filepath.removesuffix('.tif')
    if filepath.endswith('.nd2'):
        return filepath.removesuffix('.nd2')

def process_directory(root_dir, config_data):
    verbose = config_data['reader']['verbose']
    writer_data = config_data['writer']
    normalize_data, _, stitch_barcode = writer_data.values()
    print = functools.partial(builtins.print, flush=True)
    vprint = print if verbose else lambda *a, **k: None
    
    if os.path.isfile(root_dir):
        all_data = []
        file_path = root_dir
        filename = os.path.basename(file_path)
        dir_name = os.path.dirname(file_path)
        rfc_data = None
        ff_loc = os.path.join(dir_name, remove_extension(filename) + "_failed_files.txt")
        open(ff_loc, 'w').close()
        time_filepath = os.path.join(dir_name, filename + 'time.txt')
        time_file = open(time_filepath, "w", encoding="utf-8")
        time_file.write(file_path + "\n")
        start_time = time.time()
        file_count = 1
        try:
            rfc_data, file_count = execute_htp(file_path, config_data, ff_loc, file_count, total=1)
        except Exception as e:
            with open(ff_loc, "a", encoding="utf-8") as log_file:
                log_file.write(f"File: {file_path}, Exception: {str(e)}\n")
        if rfc_data == None:
            raise TypeError("Please input valid file type ('.nd2', '.tif')")
        all_data.append([filename])
        all_data.extend(rfc_data)
        all_data.append([])
        filename = remove_extension(filename) + '_'
        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time / 3600 > 1:
            elapsed_hours = int(elapsed_time // 3600)
            elapsed_minutes = (elapsed_time - (elapsed_hours * 3600))/60
            elapsed_time = f'{elapsed_hours:.2f} hours, {elapsed_minutes:.2f} minutes'
        elif elapsed_time / 60 > 1:
            elapsed_minutes = elapsed_time / 60
            elapsed_time = f'{elapsed_minutes:.2f} minutes'
        else:
            elapsed_time = f'{elapsed_time:.2f} seconds'
        vprint('Time Elapsed:', elapsed_time)
        time_file.write('Time Elapsed: ' + str(elapsed_time) + "\n")
        output_filepath = os.path.join(dir_name, filename + ' summary.csv')
        write_file(output_filepath, all_data)
        
        if stitch_barcode:
            output_figpath = os.path.join(dir_name, filename + ' summary barcode')
            gen_combined_barcode(np.array(rfc_data), output_figpath, normalize_data)

        settings_loc = os.path.join(dir_name, filename + " settings.yaml")
        with open(settings_loc, 'w+', encoding="utf-8") as ff:
            yaml.dump(config_data, ff)

        time_file.close()
        if os.stat(ff_loc).st_size == 0:
            os.remove(ff_loc)
    else: 
        all_data = []
        all_rfc_data = []
        time_filepath = os.path.join(root_dir, os.path.basename(root_dir) + ' time.txt')
        time_file = open(time_filepath, "w", encoding="utf-8")
        time_file.write(root_dir + "\n")
        
        start_folder_time = time.time()
        ff_loc = os.path.join(root_dir, "failed_files.txt")
        open(ff_loc, 'w').close()

        file_count = sum([len([file for file in files if (file.endswith(".tif") or file.endswith(".nd2"))]) for _, _, files in os.walk(root_dir)])
        file_itr = 1
        
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames]
    
            for filename in filenames:
                if filename.startswith('._'):
                    continue

                file_path = os.path.join(dirpath, filename)
                start_time = time.time()
                try:
                    rfc_data, file_itr = execute_htp(file_path, config_data, ff_loc, file_itr, file_count)
                except TypeError:
                    # for ending in ["failed_files.txt", "time.txt", ".csv", ".yaml", "Flow Field.png", "Summary Graphs.png", "Comparison.png", "Thumbs.db"]:
                    #     if file_path.endswith(ending) or 'Barcode_channel_' in file_path:
                    #         file_itr -= 1 
                    # file_itr += 1
                    continue
                except Exception as e:
                    with open(ff_loc, "a", encoding="utf-8") as log_file:
                        log_file.write(f"File: {file_path}, Exception: {str(e)}\n")
                    continue
                if rfc_data == None:
                    continue
                all_data.append([file_path])
                all_data.extend(rfc_data)
                all_data.append([])
                for result in rfc_data:
                    all_rfc_data.append(np.array(result))

                end_time = time.time()
                elapsed_time = end_time - start_time
                if elapsed_time / 3600 > 1:
                    elapsed_hours = int(elapsed_time // 3600)
                    elapsed_minutes = (elapsed_time - (elapsed_hours * 3600))/60
                    elapsed_time = f'{elapsed_hours:.2f} hours, {elapsed_minutes:.2f} minutes'
                elif elapsed_time / 60 > 1:
                    elapsed_minutes = elapsed_time / 60
                    elapsed_time = f'{elapsed_minutes:.2f} minutes'
                else:
                    elapsed_time = f'{elapsed_time:.2f} seconds'
                vprint('Time Elapsed:', elapsed_time)
                time_file.write(file_path + "\n")
                time_file.write('Time Elapsed: ' + str(elapsed_time) + "\n")
        
        output_filepath = os.path.join(root_dir, os.path.basename(root_dir) + " Summary.csv")
        try:
            write_file(output_filepath, all_data)
        except:
            counter = 1
            output_filepath = os.path.join(root_dir, os.path.basename(root_dir) + f" Summary ({counter}).csv")
            while os.path.exists(output_filepath):
                counter += 1
                output_filepath = os.path.join(root_dir, os.path.basename(root_dir) + f" Summary ({counter}).csv")
            write_file(output_filepath, all_data)
        
        if stitch_barcode:
            try:
                output_figpath = os.path.join(root_dir, os.path.basename(root_dir) + '_Summary Barcode')
                gen_combined_barcode(np.array(all_rfc_data), output_figpath, normalize_data)
            except Exception as e:
                with open(ff_loc, "a", encoding="utf-8") as log_file:
                    log_file.write(f"Unable to generate barcode, Exception: {str(e)}\n")

        end_folder_time = time.time()
        elapsed_folder_time = end_folder_time - start_folder_time
        if elapsed_folder_time / 3600 > 1:
            elapsed_hours = int(elapsed_folder_time // 3600)
            elapsed_minutes = (elapsed_folder_time - (elapsed_hours * 3600))/60
            elapsed_folder_time = f'{elapsed_hours:.2f} hours, {elapsed_minutes:.2f} minutes'
        elif elapsed_folder_time / 60 > 1:
            elapsed_minutes = elapsed_folder_time / 60
            elapsed_folder_time = f'{elapsed_minutes:.2f} minutes'
        else:
            elapsed_folder_time = f'{elapsed_folder_time:.2f} seconds'
        vprint('Time Elapsed to Process Folder:', elapsed_folder_time)
        time_file.write('Time Elapsed to Process Folder: ' + str(elapsed_folder_time) + "\n")
        
        time_file.close()
        if os.stat(ff_loc).st_size == 0:
            os.remove(ff_loc)
        settings_loc = os.path.join(root_dir, os.path.basename(root_dir) + " Settings.yaml")
        with open(settings_loc, 'w+') as ff:
            yaml.dump(config_data, ff)