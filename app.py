import streamlit as st
import os
import sys
from dotenv import load_dotenv

from TorontoCanadaChapter_CanPolicyInsight.task6_model_deployment.scripts.query_engine import index_connection, initialize_retriever, index_retrieval, load_config
# Load environment variables
load_dotenv()

# Add the scripts directory to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
scripts_dir = os.path.join(base_dir, 'scripts')
logo_path = os.path.join(base_dir, 'Canada Policy', 'TorontoCanadaChapter_CanPolicyInsight', 'task6_model_deployment', 'assets', 'logo.png')
sys.path.append(scripts_dir)

# Import functions from the query engine script


def main():
    # Create a side-by-side layout using columns
    col1, col2 = st.columns([1, 5])  # Adjust the width ratios of the columns as needed

    # Display the logo in the first column
    with col1:
        if os.path.exists(logo_path):
            st.image(logo_path, width=80)  # Adjust the width to resize the logo
        else:
            st.error(f"Logo not found at: {logo_path}")

    # Display the title in the second column
    with col2:
        st.title("Canada Policy Explorer")

    # Initialize Pinecone connection and retrieverC:\Users\agshi\Desktop\Omdena\Canada Policy\TorontoCanadaChapter_CanPolicyInsight\task6_model_deployment\configs
    config_path = os.path.join(base_dir, 'Canada Policy','TorontoCanadaChapter_CanPolicyInsight','task6_model_deployment','configs','config.yaml')
    config = load_config(config_path)
    pinecone_index = index_connection(config_path)
    retriever = initialize_retriever(pinecone_index)
    st.write("Welcome to the Canada Policy Navigator. Explore policy insights and more.")
    # User input for query
    query_text = st.text_input("Enter your query:", "")

    if st.button("Submit"):
        if query_text:
            response = index_retrieval(retriever, query_text)
            st.write(response.response) 
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
