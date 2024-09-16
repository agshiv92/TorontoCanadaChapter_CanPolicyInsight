import os
import yaml
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the base directory (one level up from the script directory)
base_dir = os.path.dirname(script_dir)

# Construct the path to the config file
config_path = os.path.join(base_dir, 'configs', 'config.yaml')

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config(config_path)
embed_model_name =config['pinecone']['dimension']
print(embed_model_name)