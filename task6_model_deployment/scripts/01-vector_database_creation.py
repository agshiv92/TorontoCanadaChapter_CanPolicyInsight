import yaml
import os
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the base directory (one level up from the script directory)
base_dir = os.path.dirname(script_dir)

# Construct the path to the config file
config_path = os.path.join(base_dir, 'configs', 'config.yaml')

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def creation_of_vector_database():
    # Load the configuration from a YAML file
    config = load_config(config_path)

    # Initialize the Pinecone client
    pc = Pinecone(
        api_key=os.getenv('PINECONE_API_KEY'))  # Ensure your API key is set in the environment variables

    # Connect to the Pinecone index
    index_name = config['pinecone']['index_name']
    dimension = config['pinecone']['dimension']
    metric = config['pinecone']['metric']
    # file_path = config['file_location']['file_path']
    cloud = config['pinecone']['cloud']
    region = config['pinecone']['region']
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,  
            spec=ServerlessSpec(
                cloud=cloud,  # Specify your preferred cloud provider
                region=region  # Specify your preferred region
            )
        )

if __name__ == "__main__":
    creation_of_vector_database()

