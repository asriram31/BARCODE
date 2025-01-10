import sys, yaml
from barcoder import process_directory
from gooey import Gooey, GooeyParser
from writer import generate_aggregate_csv

@Gooey(program_name="BARCODE Program", tabbed_groups=True, navigation='Tabbed')
def main():
    parser = GooeyParser(description='Code that runs through the BARCODE code developed by the DMREF group')

    gc = parser.add_argument_group("Execution Settings")

    # File/Directory Selection
    fdc = gc.add_mutually_exclusive_group()
    fdc.add_argument('--file_path', metavar = 'File Selection', widget='FileChooser', gooey_options = {
        'wildcard': "Document (*.nd2)|*.nd2|"
        "TIFF Image (*.tif)|*.tif"
    })
    fdc.add_argument('--dir_path', metavar='Folder Selection', widget='DirChooser')
    # Channel Selection
    c_select = gc.add_mutually_exclusive_group()
    c_select.add_argument('--channels', metavar='Parse All Channels', widget='CheckBox', action='store_true')
    c_select.add_argument('--channel_selection', metavar='Choose Channel', widget='IntegerField', gooey_options = {
        'min': -3, 
        'max': 4
    })
    fdc.add_argument('--barcode_generation', metavar='Combine CSV files/Generate Barcodes', help='Click to combine summary CSV files and generate barcodes', widget='CheckBox', action='store_true')

    # Reader Execution Settings
    gc.add_argument('--check_resilience', metavar='Binarization', help='Evaluate sample(s) using binarization module', widget='CheckBox', action='store_true')
    gc.add_argument('--check_flow', metavar='Optical Flow', help='Evaluate sample(s) using optical flow module', widget='CheckBox',  action='store_true')
    gc.add_argument('--check_coarsening', metavar='Intensity Distribution', help='Evaluate sample(s) using intensity distribution module', widget='CheckBox', action='store_true')
    gc.add_argument('--dim_images', metavar='Include Dim Files', help='Click to scan files that may be too dim to accurately profile', widget='CheckBox', action='store_true')
    gc.add_argument('--dim_channels', metavar='Include Dim Channels', help='Click to scan channels that may be too dim to accurately profile', widget='CheckBox', action='store_true')

    # Writer Data
    gc.add_argument('--verbose', metavar='Verbose', help='Show more details', widget='CheckBox', action='store_true')
    gc.add_argument('--return_graphs', metavar='Save Graphs', help='Click to save graphs representing sample changes', widget='CheckBox', action='store_true')
    gc.add_argument('--return_intermediates', metavar='Save Intermediates', help='Click to save intermediate data structures (flow fields, binarized images, intensity distributions)', widget='CheckBox', action='store_true')
    
    gc.add_argument('--stitch_barcode', metavar='Dataset Barcode', help="Generates an aggregate barcode for the dataset", widget="CheckBox", action='store_true')
    gc.add_argument('--normalize_data', metavar='Normalize Dataset Barcode', help='Uses the dataset to generate a normalized aggregate barcode for the dataset', widget='CheckBox', action='store_true')

    gc.add_argument('--configuration_file', metavar='Configuration YAML File', help="Load a preexisting configuration file for the settings", widget="FileChooser", gooey_options = {
        'wildcard': "YAML (*.yaml)|*.yaml|"
        "YAML (*.yml)|*.yml"
    })


    res_settings = parser.add_argument_group('Binarization Settings')
    res_settings.add_argument('--r_offset', metavar='Binarization Threshold', help='Adjust the binarization threshold as a percentage of the mean (calculated as (1 + offset) * mean)', widget='DecimalField', default=0.1, gooey_options = {
        'min':-1.0,
        'max':1.0,
        'increment':0.05
    })

    res_settings.add_argument('--res_f_step', metavar = 'Frame Step', help = "Controls the interval between binarized frames", widget='Slider', default=10, gooey_options = {
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

    flow_settings = parser.add_argument_group('Optical Flow Settings')

    flow_settings.add_argument('--flow_f_step', metavar = 'Frame Step', help = "Controls the interval between frames the flow field is calculated at", widget = 'Slider', default = 10, gooey_options = {
        'min':1,
        'increment':1
    })
    
    flow_settings.add_argument('--win_size', metavar = 'Window Size', help = "Controls the window size for the optical flow field estimation", widget = 'IntegerField', default = 32, gooey_options = {
        'min':1,
        'increment':1
    })

    flow_settings.add_argument('--downsample', metavar = 'Downsample', help = "Controls the downsampling rate of the flow field (larger values give less precision, less prone to noise)", widget = 'IntegerField', default = 8, gooey_options = {
        'min':1,
        'increment':1
    })
    
    flow_settings.add_argument('--nm_pixel_ratio', metavar = 'Nanometer to Pixel Ratio', help = "Set the ratio of nanometers to pixels (leave at default if this is variable within your dataset)", widget= 'IntegerField', default = 1, gooey_options = {
        'min':1,
        'increment':1,
        'max': 10 ** 6
    })
    
    flow_settings.add_argument('--frame_interval', metavar = 'Frame Interval', help = "Set the interval (in seconds) between frames (leave at default if this is variable within your dataset", widget= 'IntegerField', default = 1, gooey_options = {
        'min':1,
        'increment':1,
        'max': 10 ** 3
    })
    

    coarse_settings = parser.add_argument_group('Intensity Distribution Settings')

    coarse_settings.add_argument('--first_frame', metavar='First Frame', help = 'Controls which frame is used as the first frame for intensity distribution comparisons', widget='IntegerField', gooey_options = {
        'min':1,
        'increment':1
    })

    coarse_settings.add_argument('--last_frame', metavar = 'Last Frame', help = "Select which frame is used as the second frame for intensity distribution comparisons (0 for the final frame of video)", widget = 'IntegerField', default=0, gooey_options = {
        'min':0,
        'increment':1
    })

    coarse_settings.add_argument('--pf_evaluation', metavar = 'Percent of Frames Evaluated', help = "Determine what fraction of frames are evaluated for intensity distribution comparison using mean-mode comparison", widget = 'DecimalField', default = 0.1, gooey_options = {
        'min':0.01,
        'max': 0.2,
        'increment':0.01
    })

    barcode_generator = parser.add_argument_group('Barcode Generator + CSV Aggregator')
    barcode_generator.add_argument('--csv_paths', metavar = 'CSV File Locations', widget='MultiFileChooser', help="Select the CSV files representing the datasets you would like to combine", gooey_options = {
        'wildcard': "CSV Document (*.csv)|*.csv"})
    barcode_generator.add_argument('--combined_location', metavar = 'Aggregate Location', widget='FileSaver', help="Select a location for the aggregate CSV file to be located", gooey_options = {
        'default_file': "aggregate_summary.csv"
    })
    barcode_generator.add_argument('--generate_agg_barcode', metavar = 'Generate Aggregate Barcode', widget='CheckBox', help="Click to generate an aggregate barcode from these files", action="store_true")
    barcode_generator.add_argument('--normalize_agg_barcode', metavar = 'Normalize Aggregate Barcode', widget='CheckBox', help="Click to normalize the barcode (color will be determined by the limits of the dataset)", action='store_true')

    headers = ['Default', 'Connectivity', 'Maximum Island Area', 'Maximum Void Area', 
            'Void Area Change', 'Island Area Change', 'Initial Island Area 1', 
            'Initial Island Area 2', 'Maximum Kurtosis', 'Maximum Median Skewness', 
            'Maximum Mode Skewness', 'Kurtosis Difference', 'Median Skewness Difference', 
            'Mode Skewness Difference', 'Mean Speed', 'Speed Change',
            'Mean Flow Direction', 'Flow Directional Spread']

    barcode_generator.add_argument('--sort', metavar = 'Parameter Sort', widget = 'Dropdown', help='Select a parameter to sort the barcode on', choices = headers)
    
    
    settings = parser.parse_args()
    
    if (settings.barcode_generation):
        files = settings.csv_paths.split(',')
        combined_csv_loc = settings.combined_location

        gen_barcode = settings.generate_agg_barcode
        normalize_data = settings.normalize_agg_barcode
        sort_param = None if settings.sort == 'Default' else settings.sort
        generate_aggregate_csv(files, combined_csv_loc, gen_barcode, normalize_data, sort_param)
        
    else:
        if not (settings.dir_path or settings.file_path):
            print("No file or directory has been selected, exiting the program...")
            sys.exit()

        if not (settings.channels or settings.channel_selection):
            print("No channel has been specified, exiting the program...")
            sys.exit()

        dir_name = settings.dir_path if settings.dir_path else settings.file_path

        if settings.configuration_file:
            with open(settings.configuration_file, 'r') as f:
                config_data = yaml.load(f, Loader=yaml.FullLoader)
                # if config_data['reader']['channel_select'] == 'All':
                #     config_data['reader']['channel_select']

        else: 
            config_data = set_config_data(settings)

        print(dir_name, flush = True)

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
            'accept_dim_channels':args.dim_channels,
            'accept_dim_images':args.dim_images,
            'channel_select':'All' if args.channels else int(args.channel_selection),
            'coarsening':args.check_coarsening,
            'flow':args.check_flow,
            'resilience':args.check_resilience,
            'return_graphs':args.return_graphs,
            'verbose':args.verbose
        }
        
        writer_data = {
            'normalize_data':args.normalize_data,
            'return_intermediates':args.return_intermediates,
            'stitch_barcode':args.stitch_barcode
        }
        
        if reader_data['resilience']:
            resilience_data = {
                'evaluation_settings':{
                    'f_start':float(args.pf_start),
                    'f_stop':float(args.pf_stop)
                },
                'frame_step':int(args.res_f_step),
                'r_offset':float(args.r_offset),
            }
        if reader_data['flow']:
            flow_data = {
                'downsample':int(args.downsample),
                'frame_step':int(args.flow_f_step),
                'frame_interval':int(args.frame_interval),
                'nm_pixel_ratio':int(args.nm_pixel_ratio),
                'win_size':int(args.win_size)
            }

        if reader_data['coarsening']:
            coarsening_data = {
                'evaluation_settings':{
                    'first_frame':int(args.first_frame), 
                    'last_frame':False if int(args.last_frame) == 0 else int(args.last_frame)
                },
                'mean_mode_frames_percent':float(args.pf_evaluation),
            }

        config_data = {
            'coarse_parameters':coarsening_data,
            'flow_parameters':flow_data,
            'reader':reader_data,
            'resilience_parameters':resilience_data,
            'writer':writer_data
        }
        
    return config_data

if __name__ == "__main__":
    main()
