from gooey import Gooey, GooeyParser
import main

@Gooey()
def foobar():
    parser_a = GooeyParser(description='Code that runs through the BARCODE code developed by the DMREF group')

    gc = parser_a.add_argument_group("Execution Settings")
    fdc = gc.add_mutually_exclusive_group()
    fdc.add_argument('--file_path', metavar = 'File Chooser', widget='FileChooser', gooey_options = {
        'wildcard':
            "*.nd2|"
            "*.tiff|"
            "*.tif"
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

    gc.add_argument('--check_resilience', metavar='Resilience', help='Evaluate sample(s) using binarization module', widget='CheckBox')
    gc.add_argument('--check_flow', metavar='Flow', help='Evaluate sample(s) using optical flow module', widget='CheckBox')
    gc.add_argument('--check_coarsening', metavar='Coarsening', help='Evaluate sample(s) using intensity distribution module', widget='CheckBox')
    
    gc.add_argument('--Verbose', metavar='Verbose', help='Show more details', widget='CheckBox')
    gc.add_argument('--return_graphs', metavar='Save Graphs', help='Click to save graphs representing sample changes', widget='CheckBox')
    gc.add_argument('--return_intermediates', metavar='Intermediates', help='Click to save intermediate data structures (flow fields, binarized images, intensity distributions)', widget='CheckBox')

    gc.add_argument('--dim_images', metavar='Dim Images', help='Click to scan files that may be too dim to accurately profile', widget='CheckBox')
    gc.add_argument('--dim_channels', metavar='Dim Channels', help='Click to scan channels that may be too dim to accurately profile', widget='CheckBox')

    settings = parser_a.parse_args()
    main.main()

if __name__ == "__main__":
    foobar()