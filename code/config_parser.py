
import argparse
import yaml

def parse_config(config_file, cli_args=None):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    if cli_args:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_type", type=str, help="Type of foundation model")
        parser.add_argument("--epochs", type=int, help="Number of training epochs")
        # Add more arguments as needed
        args = vars(parser.parse_args(cli_args))
        config.update({k: v for k, v in args.items() if v is not None})
    return config
