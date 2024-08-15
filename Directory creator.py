import os

# Define the base directory where you want to create the project structure
base_dir = r"C:\Users\agshi\Desktop\Omdena\Canada Policy\TorontoCanadaChapter_CanPolicyInsight"

# List of folders to be created
folders = [
    "task0_learning_and_research",
    "task1_data_collection",
    "task2_eda_preprocessing",
    "task3_model_building",
    "task4_prompt_engg",
    "task5_model_evaluation",
    "task6_model_deployment",
    "task7_final_presentation"
]

# Function to create the directory structure
def create_folder_structure(base_dir, folders):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    for folder in folders:
        path = os.path.join(base_dir, folder)
        os.makedirs(path, exist_ok=True)
        print(f"Created: {path}")

# Create the folder structure
create_folder_structure(base_dir, folders)

print("Folder structure created successfully!")