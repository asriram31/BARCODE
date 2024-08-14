import os, pims, functools, builtins
import imageio.v3 as iio
import numpy as np
from nd2reader import ND2Reader

def read_file(file_path, accept_dim = False):
    print = functools.partial(builtins.print, flush=True)
    acceptable_formats = ('.tiff', '.tif', '.nd2')
    if (os.path.exists(file_path) and file_path.endswith(acceptable_formats)) == False:
        return None

    def check_first_frame_dim(file):
        min_intensity = np.min(file[0])
        mean_intensity = np.mean(file[0])
        return 2 * np.exp(-1) * mean_intensity <= min_intensity

    def bleach_correction(im):
        min_px_intensity = np.min(im)
        num_frames=len(im) #num frames in video
        corrected_frames = np.zeros_like(im)
        
        # Calculate the mean intensity of the first frame
        i_frame_data = im[0] - min_px_intensity
        initial_mean_intensity = np.mean(i_frame_data)
        
        #bleach correction for each frame
        for i in range(0, num_frames):
            frame_data=im[i] - min_px_intensity
            # Calculate normalization factor relative to the first frame
            normalization_factor=initial_mean_intensity / np.mean(frame_data)
            corrected_frames[i] = normalization_factor * frame_data + min_px_intensity
        return corrected_frames

    
    def convert_to_array(file):
        num_images = file.sizes['t']
        num_channels = len(file.metadata['channels'])
        height = file.metadata['height']
        width = file.metadata['width']
        images = np.zeros((num_images, height, width, num_channels))

        if num_images <= 1: # Checks to see if file is z-stack instead of time series
            return None
        
        for i in range(num_channels):
            for j in range(num_images):
                frame = np.array(file.get_frame_2D(c=i, t=j))
                images[j, :, :, i] = frame
                
        return images
    print(file_path)
    if file_path.endswith('.tiff') or file_path.endswith('.tif'):
        file = iio.imread(file_path)
        file = np.reshape(file, (file.shape + (1,))) if len(file.shape) == 3 else file

    elif file_path.endswith('.nd2'):
        try:
            file_nd2 = ND2Reader(file_path)
            if file_nd2 == None:
                return None
        except:
            return None
        file = convert_to_array(file_nd2)
        if isinstance(file, np.ndarray) == False:
            return None

    # file = bleach_correction(file)

    if len(file.shape) == 2:
        print("Static image: can not capture dynamics, skipping to next file...")

    if (file == 0).all():
        print('Empty file: can not process, skipping to next file...')
        return None
    
    if accept_dim == False and check_first_frame_dim(file) == True:
        print(file_path + 'is too dim, skipping to next file...')
        return None
        
    else:
        return file
