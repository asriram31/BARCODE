import os, functools, builtins
import imageio.v3 as iio
import numpy as np
import nd2

def read_file(file_path, count, total_count, accept_dim = False, allow_large_files = True):
    print = functools.partial(builtins.print, flush=True)
    acceptable_formats = ('.tif', '.nd2')
    accept_endings = ["failed_files.txt", "time.txt", ".csv", ".yaml", "Flow Field.png", "Summary Graphs.png", "Comparison.png", "Thumbs.db"]
    for ending in accept_endings:
        if file_path.endswith(ending) or 'Barcode_channel_' in file_path:
            return None
    
    print(f'File {count} of {total_count}')
    print(file_path)
    if (os.path.exists(file_path) and file_path.endswith(acceptable_formats)) == False:
        print("Invalid format: files must be .tif or .nd2.")
        return None

    file_size = os.path.getsize(file_path)
    file_size_gb = file_size / (1024 ** 3)
    if file_size_gb > 5 and not allow_large_files:
        print("File size is too large -- this program does not process files larger than 5 GB.")
        return None

    def check_first_frame_dim(file):
        min_intensity = np.min(file[0])
        mean_intensity = np.mean(file[0])
        return 2 * np.exp(-1) * mean_intensity <= min_intensity
    
    if file_path.endswith('.tif'):
        file = iio.imread(file_path)
        file = np.reshape(file, (file.shape + (1,))) if len(file.shape) == 3 else file

    elif file_path.endswith('.nd2'):
        try:
            with nd2.ND2File(file_path) as ndfile:
                if len(ndfile.shape) >= 5 or "Z" in ndfile.sizes:
                    print('Z-stack identified, skipping to next file...')
                    return None
                if 'T' not in ndfile.sizes or len(ndfile.shape) <= 2 or ndfile.sizes['T'] <= 5:
                    print('Too few frames, unable to capture dynamics, skipping to next file...')
                    return None
                if ndfile == None:
                    print('No file detected, skipping to next file...')
                    return None
                file = ndfile.asarray()
                shape = (file.shape[0], file.shape[2], file.shape[3], file.shape[1]) # Reorder
                file = np.swapaxes(np.swapaxes(file, 1, 2), 2, 3)

        except Exception as e:
            print(f'Unable to read file, skipping to next file...')
            return None
        if isinstance(file, np.ndarray) == False:
            return None
        
    if (file == 0).all():
        print('Empty file: can not process, skipping to next file...')
        return None
    
    if accept_dim == False and check_first_frame_dim(file) == True:
        print(file_path + 'is too dim, skipping to next file...')
        return None
        
    else:
        return file
