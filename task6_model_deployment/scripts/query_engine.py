import os
import yaml
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.response.pprint_utils import pprint_source_node
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# Load environment variables from the .env file
load_dotenv()

# Function to load YAML configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Pinecone Index Connection
def index_connection(config_path):
    """
    Initializes the Pinecone client and retrieves the index using the provided YAML configuration.
    
    Args:
    config_path (str): Path to the YAML configuration file.
    
    Returns:
    index: The initialized Pinecone index.
    """
    # Load the configuration from a YAML file
    config = load_config(config_path)
    embed_model_name = config['embeddings']['model_name']
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    model_name = config['model']['model_name']
    Settings.llm = Groq(model=model_name, api_key=os.getenv('GROQ_API_KEY'))
    Settings.embed_model = embed_model   
    # Initialize the Pinecone client
    pc = Pinecone(
        api_key=os.getenv('PINECONE_API_KEY')  # Get the Pinecone API key from the environment
    )
    index_name = config['pinecone']['index_name']
    index = pc.Index(index_name)  # Get the Pinecone index using the index name from the config
    return index

# Initialize Pinecone Vector Store and Retriever
def initialize_retriever(pinecone_index):
    """
    Initializes the Pinecone vector store and sets up the retriever.
    
    Args:
    pinecone_index: The Pinecone index object.
    
    Returns:
    retriever: The initialized retriever for querying the vector store.
    """

    # Initialize Pinecone Vector Store
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index, text_key="_node_content")
    
    # Create the retriever using the VectorStoreIndex and configure similarity_top_k
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    return index

# Query the Pinecone Index
def index_retrieval(index, query_text):
    """
    Queries the Pinecone index using the provided retriever and query text.
    
    Args:
    retriever: The initialized retriever.
    query_text (str): The text query to search for.
    
    Returns:
    str: Query result from the Pinecone index.
    """

    # Execute the query using the retriever
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)

    print(response)# Pretty print the source node for clarity
    
    return response

# Example usage
if __name__ == "__main__":
    # Dynamically determine the path to the config file
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current script directory
    base_dir = os.path.dirname(script_dir)  # Go one level up
    config_path = os.path.join(base_dir, 'configs', 'config.yaml')  # Path to 'config.yaml' in the 'configs' directory
    
    # Step 1: Initialize Pinecone Connection
    pinecone_index = index_connection(config_path=config_path)
    
    # Step 2: Initialize the Retriever
    retriever = initialize_retriever(pinecone_index)
    
    # Step 3: Query the Pinecone index
    query_text = """How much can the Minister of Health pay out of the Consolidated Revenue Fund in relation to coronavirus disease 2019 (COVID-19) tests"""
    response = index_retrieval(retriever, query_text)
    
    # Print the result (already printed by pprint_source_node)
