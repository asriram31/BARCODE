import imageio.v3 as iio
import pims
from nd2reader import ND2Reader
import os, csv, sys
from resilience_tracker import check_resilience
from flow_tracker import check_flow
from coarse_tracker import check_coarse
import numpy as np


def read_file(file_path):
    acceptable_formats = ('.tiff', '.tif', '.nd2')
    if (os.path.exists(file_path) and file_path.endswith(acceptable_formats)) == False:
        return None
    
    def convert_to_array(file):
        num_images = file.sizes['t']
        num_channels = file.sizes['c']
        height = file.metadata['height']
        width = file.metadata['width']
        images = np.zeros((num_images, height, width, num_channels))
        for i in range(num_channels):
            for j in range(num_images):
                frame = np.array(file.get_frame_2D(c=i, t=j))
                images[j, :, :, i] = frame
        return images
    
    if file_path.endswith('.tiff') or file_path.endswith('.tif'):
        file = iio.imread(file_path)
        if len(file.shape) == 3:
            file = np.reshape(file, (file.shape + (1,)))
        channels = file.shape[3]
        filetype = 'tif'

    elif file_path.endswith('.nd2'):
        file_nd2 = ND2Reader(file_path)
        file = convert_to_array(file_nd2)
        channels = len(file_nd2.metadata['channels'])

    return file

def execute_htp(filepath, channel_select=-1, resilience=True, flow=True, coarse=True, verbose=False):
    def check(channel, resilience, flow, coarse):
        if resilience == True:
            r, rfig = check_resilience(file, channel)
        else:
            r = "Resilience not tested"
        if flow == True:
            f, ffig = check_flow(file, channel)
        else:
            f = "Flow not tested"
        if coarse == True:
            c, cfig = check_coarse(file, channel)
        else:
            c = "Coarseness not tested."
        return [channel, r, f, c]
    
    file = read_file(filepath)

    if (isinstance(file, np.ndarray) == False):
        return None

    channels = file.shape[3]
    
    if (isinstance(channel_select, int) == False) or channel_select > channels:
        raise ValueError("Please give correct channel input (0 for all channels, 1 for channel 1, etc)")
    
    rfc = []
    
    if channel_select == -1:
        for channel in range(channels):
            print('Channel:', channel)
            rfc.append(check(channel, resilience, flow, coarse))
    
    else: 
        rfc.append(check(channel_select, resilience, flow, coarse))

    return rfc

def process_directory(root_dir):

    if os.path.isfile(root_dir):
        all_data = []
        file_path = root_dir
        filename = os.path.basename(file_path)
        dir_name = os.path.dirname(file_path)
        rfc_data = execute_htp(file_path)
        if rfc_data == None:
            raise TypeError("Please input valid file type ('.nd2', '.tiff', '.tif')")
        all_data.append([filename])
        all_data.extend(rfc_data)
        all_data.append([])

        if all_data:
            headers = ['Channel', 'Resilience', 'Flow', 'Coarseness']
            output_filepath = os.path.join(dir_name, "summary.csv")
            with open(output_filepath, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                for entry in all_data:
                    if isinstance(entry, list) and len(entry) == 1:
                        # Write the file name
                        csvwriter.writerow(entry)
                    elif entry:
                        # Write the headers if entry contains channel data
                        csvwriter.writerow(headers)
                        headers = []  # Ensure headers are only written once per file
                        csvwriter.writerow(entry)
                    else:
                        # Write an empty row
                        csvwriter.writerow([])
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        all_data = []

        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            print(file_path)
            rfc_data = execute_htp(file_path)
            if rfc_data == None:
                continue
            all_data.append([filename])
            all_data.extend(rfc_data)
            all_data.append([])

        if all_data:
            headers = ['Channel', 'Resilience', 'Flow', 'Coarseness']
            output_filepath = os.path.join(dirpath, "summary.csv")
            with open(output_filepath, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                for entry in all_data:
                    if isinstance(entry, list) and len(entry) == 1:
                        # Write the file name
                        csvwriter.writerow(entry)
                    elif entry:
                        # Write the headers if entry contains channel data
                        csvwriter.writerow(headers)
                        headers = []  # Ensure headers are only written once per file
                        csvwriter.writerow(entry)
                    else:
                        # Write an empty row
                        csvwriter.writerow([])

def main():
    dir_name = sys.argv[1]
    process_directory(dir_name)

if __name__ == "__main__":
    main()