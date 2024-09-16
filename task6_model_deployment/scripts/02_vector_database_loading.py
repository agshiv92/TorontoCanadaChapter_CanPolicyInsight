import yaml
import os
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
from dotenv import load_dotenv
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.groq import Groq


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

def index_connection():
    # Load the configuration from a YAML file
    config = load_config(config_path)

    # Initialize the Pinecone client
    pc = Pinecone(
        api_key=os.getenv('PINECONE_API_KEY')
    )
    index_name = config['pinecone']['index_name']
    index = pc.Index(index_name)
    return index

def chunk_documents(directory_path="./data/paul_graham"):
    """
    Reads documents from a specified directory and chunks them.
    
    Args:
    directory_path (str): The path of the directory containing documents to read.
    
    Returns:
    List[Document]: A list of document chunks that will be indexed.
    """
    # Load documents from the directory
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    # Here you could apply further chunking logic if needed (for example, split large documents into smaller chunks)
    # For now, we're assuming the reader does basic chunking for us
    
    return documents

# Part 2: Loading Chunks into Pinecone
def load_chunks_into_pinecone(documents):
    config = load_config(config_path)
    pinecone_index = index_connection()
    model_name = config['model']['model_name']
    embed_model_name = config['embeddings']['model_name']
    print(embed_model_name)
    Settings.llm = Groq(model=model_name, api_key=os.getenv('GROQ_API_KEY'))
    Settings.chunk_size = config['pinecone']['dimension']
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    Settings.embed_model = embed_model

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    # Create the storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create the index with the documents
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    
    print("Data has been successfully loaded into the Pinecone index!")
    
    return index

# Example usage
if __name__ == "__main__":

    # Step 1: Chunk the documents
    documents = chunk_documents(directory_path=r"C:\Users\agshi\Desktop\Omdena\Canada Policy\TorontoCanadaChapter_CanPolicyInsight\task6_model_deployment\assets")

    # Step 2: Load the chunks into Pinecone
    index = load_chunks_into_pinecone(documents)
