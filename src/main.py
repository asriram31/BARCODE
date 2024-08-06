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
from barcoder import process_directory, execute_htp
from gooey import Gooey, GooeyParser

@Gooey(program_name="DMREF BARCODE Program", tabbed_groups=True, navigation='Tabbed')
def main():
    parser = GooeyParser(description='Code that runs through the BARCODE code developed by the DMREF group')

    gc = parser.add_argument_group("Execution Settings")

    # File/Directory Selection
    fdc = gc.add_mutually_exclusive_group(required = True)
    fdc.add_argument('--file_path', metavar = 'File Chooser', widget='FileChooser', gooey_options = {
        'wildcard': "Document (*.nd2)|*.nd2|"
        "TIFF Image (*.tiff)|*.tiff|"
        "TIFF Image (*.tif)|*.tif"
    })
    fdc.add_argument('--dir_path', metavar='Directory Chooser', widget='DirChooser')

    gc.add_argument('--ignored_dirs', metavar='Skip Directories', widget='MultiDirChooser', help="Select directories for the program to ignore")
    # Channel Selection
    c_select = gc.add_mutually_exclusive_group(required = True)
    c_select.add_argument('--channels', metavar='Parse All Channels', widget='CheckBox', action='store_true')
    c_select.add_argument('--channel_selection', metavar='Choose Channel', widget='IntegerField', gooey_options = {
        'min': -3, 
        'max': 4
    })

    # Reader Execution Settings
    gc.add_argument('--check_resilience', metavar='Resilience', help='Evaluate sample(s) using binarization module', widget='CheckBox', action='store_true')
    gc.add_argument('--check_flow', metavar='Flow', help='Evaluate sample(s) using optical flow module', widget='CheckBox',  action='store_true')
    gc.add_argument('--check_coarsening', metavar='Coarsening', help='Evaluate sample(s) using intensity distribution module', widget='CheckBox', action='store_true')
    gc.add_argument('--dim_images', metavar='Dim Images', help='Click to scan files that may be too dim to accurately profile', widget='CheckBox', action='store_true')
    gc.add_argument('--dim_channels', metavar='Dim Channels', help='Click to scan channels that may be too dim to accurately profile', widget='CheckBox', action='store_true')

    # Writer Data
    gc.add_argument('--verbose', metavar='Verbose', help='Show more details', widget='CheckBox', action='store_true')
    gc.add_argument('--return_graphs', metavar='Save Graphs', help='Click to save graphs representing sample changes', widget='CheckBox', action='store_true')
    gc.add_argument('--return_intermediates', metavar='Intermediates', help='Click to save intermediate data structures (flow fields, binarized images, intensity distributions)', widget='CheckBox', action='store_true')
    gc.add_argument('--normalize_dataset', metavar='Dataset Normalization', help="Click to generate the barcode colormaps using normalization determined by the dataset", widget='CheckBox', action='store_true')
    
    gc.add_argument('--generate_rgb_map', metavar='Generate RGB Map', help="Click to output the RGB map", widget="CheckBox", action='store_true')
    gc.add_argument('--generate_barcode', metavar='Generate Barcode', help="Click to create barcodes for the dataset", widget="CheckBox", action='store_true')
    gc.add_argument('--stitch_barcode', metavar='Dataset Barcode', help="Generates a barcode for the entire dataset, instead of for individual videos (only occurs if barcode is generated)", widget="CheckBox", action='store_true')


    res_settings = parser.add_argument_group('Resilience Settings')
    res_settings.add_argument('--r_offset', metavar='Binarization Threshold', help='Adjust the pixel intensity threshold as a percentage of the mean (0 - 200%)', widget='DecimalField', default=0.1, gooey_options = {
        'min':-1.0,
        'max':1.0,
        'increment':0.05
    })
    res_settings.add_argument('--pt_loss', metavar='Percent Threshold Loss', help="Percentage of original void size that final void size must be greater than to considered resilient", widget='DecimalField', default = 0.9, gooey_options = {
        'min':0,
        'max':1,
        'increment':0.05 
    })
    res_settings.add_argument('--pt_gain', metavar='Percent Threshold Gain', help="Percentage of original void size that final void size must be less than to considered resilient", widget='DecimalField', default = 1.1, gooey_options = {
        'min':1,
        'max':5,
        'increment':0.05 
    })

    res_settings.add_argument('--res_f_step', metavar = 'Frame Step', help = "Controls how many frames between evaluated frames", widget='Slider', gooey_options = {
        'min':1,
        'increment':1
    })

    res_settings.add_argument('--pf_start', metavar='Frame Start Percent', help="Determines starting percentage of frames to evaluate for resilience", widget='DecimalField', default = 0.9, gooey_options = {
        'min':0.5,
        'max':0.9,
        'increment':0.05
    })

    res_settings.add_argument('--pf_stop', metavar='Frame Stop Percent', help="Determines ending percentage of frames to evaluate for resilience", widget='DecimalField', default = 1, gooey_options = {
        'min':0.9,
        'max':1,
        'increment':0.05
    })

    flow_settings = parser.add_argument_group('Flow Settings')

    flow_settings.add_argument('--flow_f_step', metavar = 'Frame Step', help = "Controls the interval between frames the flow field is calculated at", widget = 'Slider', default = 40, gooey_options = {
        'min':1,
        'increment':1
    })

    flow_settings.add_argument('--downsample', metavar = 'Downsample', help = "Controls the downsampling rate of the flow field (larger values give less precision, less prone to noise)", widget = 'IntegerField', default = 8, gooey_options = {
        'min':1,
        'increment':1,
    })

    coarse_settings = parser.add_argument_group('Coarsening Settings')

    coarse_settings.add_argument('--first_frame', metavar='First Frame', help = 'Controls which frame is used as the first frame for intensity distribution comparisons', widget='Slider', gooey_options = {
        'min':1,
        'increment':1
    })

    final_frame = coarse_settings.add_mutually_exclusive_group(required = True)

    final_frame.add_argument('--eval_last_frame', metavar = 'Use Default Last Frame', help = 'Use final frame of image for intensity distribution comparisons', widget='CheckBox', action='store_true')

    final_frame.add_argument('--select_last_frame', metavar = 'Select Last Frame', help = "Select the final frame of the video for intensity distribution comparisons", widget = 'IntegerField')

    coarse_settings.add_argument('--thresh_percent', metavar = 'Threshold Percentage', help = 'Select the threshold percentage mean-mode difference between the final and initial intensity distributions; adjust for different objective lenses (5-7 for 60x objective lens, 1 for 20x, etc)', widget = 'Slider', default = 6, gooey_options = {
        'min': 1,
        'max': 8,
        'increment': 1
    })

    coarse_settings.add_argument('--pf_evaluation', metavar = 'Percent of Frames Evaluated', help = "Determine what percent of frames are evaluated for intensity distribution comparison using mean-mode comparison", widget = 'DecimalField', default = 0.1, gooey_options = {
        'min':0.01,
        'max': 0.2,
        'increment':0.01
    })
    
    settings = parser.parse_args()

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
            'return_graphs':args.return_graphs,
            'ignored_directories':args.ignored_dirs,
            'accept_dim_images':args.dim_images,
            'accept_dim_channels':args.dim_channels
        }
        
        writer_data = {
            'return_intermediates':args.return_intermediates,
            'normalize_dataset':args.normalize_dataset,
            'generate_rgb_map':args.generate_rgb_map,
            'generate_barcode':args.generate_barcode,
            'stitch_barcode':args.stitch_barcode
        }
        
        if reader_data['resilience']:
            resilience_data = {
                'r_offset':float(args.r_offset),
                'percent_threshold':{
                    'pt_loss':float(args.pt_loss), 
                    'pt_gain':float(args.pt_gain)
                },
                'frame_step':int(args.res_f_step),
                'evaluation_settings':{
                    'f_start':float(args.pf_start),
                    'f_stop':float(args.pf_stop)
                },
            }
        if reader_data['flow']:
            flow_data = {
                'frame_step':int(args.flow_f_step),
                'downsample':int(args.downsample)
            }

        if reader_data['coarsening']:
            coarsening_data = {
                'evaluation_settings':{
                    'first_frame':int(args.first_frame), 
                    'last_frame':False if args.eval_last_frame else args.select_last_frame
                },
                'threshold_percentage':float(args.thresh_percent),
                'mean_mode_frames_percent':float(args.pf_evaluation),
            }

        config_data = {
            'reader':reader_data,
            'resilience_parameters':resilience_data,
            'flow_parameters':flow_data,
            'coarse_parameters':coarsening_data,
            'writer':writer_data
        }
        
    return config_data

if __name__ == "__main__":
    main()