import yaml
import argparse


def read_yaml_config(file_path, default_config):
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        return {**default_config, **config}  # Overwrite default values with provided values
    except FileNotFoundError:
        return default_config

def show_default_config(default_config):
    print("Default Configuration:")
    for key, value in default_config.items():
        print(f"{key}: {value}")


def parse_sceo_command_line_args(default_config):
    parser = argparse.ArgumentParser(description='sceodesic pipeline')
    parser.add_argument('--config', type=str, default='', help='Path to YAML configuration file.')
    
    # -1 means not set, use default
    parser.add_argument('--num_hvg', type=int, help='Number of highly variable genes.', default=-1)  
    
    parser.add_argument('--action', type=int, help='Actions to control the pipeline.', default=1)
    parser.add_argument('--inp_data', type=str, required=True, help='Path to input data.')
    parser.add_argument('--output_prefix', type=str, required=True, help='Output prefix for file names.', default='DEFAULT')
    parser.add_argument('--filepath', type=str, help="Path in which to store results. Note that this argument will overwrite the file path specified in the config file. Required if config file is NOT provided.")
    parser.add_argument('--adata_output_name', type=str, help="The path of the file containing the adata embedding output, will overwrite the default path.")
    args = parser.parse_args()
    
    # Read from YAML config if provided
    if args.config:
        cfg = read_yaml_config(args.config, default_config)
        cfg['output_prefix'] =  cfg.get('output_prefix', args.output_prefix)
        
        # by default command line argument is set to -1
        cfg['num_hvg'] = args.num_hvg if args.num_hvg > 0 else cfg['num_hvg']
        args.config = cfg
    else:
        args.config = default_config

        # if config file is not provided, we must specify a file path
        if not args.filepath:
            message = ("ArgumentError: You must either pass in a config.yaml file"
                       " or an output directory using the `--filepath` argument")
            sys.exit(message)

    # note that args.filepath will overwrite the filepath specified in the config file
    if args.filepath:
        args.config['filepath'] = args.filepath

    return args