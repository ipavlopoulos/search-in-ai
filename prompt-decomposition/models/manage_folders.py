# manage_folders.py
import os
import json

def save_results_to_json(results, folder, filename):
    """Saves a list of dictionaries to a JSON file."""
    os.makedirs(folder, exist_ok=True)
    full_path = os.path.join(folder, filename)
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved successfully to '{full_path}'")


def dataset_folders(base_results_folder, dataset_path, few_shot_examples_path):
    zero_shot_folder = os.path.join(base_results_folder, 'zero_shot') # Zero-shot predictions
    few_shot_folder = os.path.join(base_results_folder, 'few_shot')  # Few-shot predictions
    os.makedirs(zero_shot_folder, exist_ok=True)
    os.makedirs(few_shot_folder, exist_ok=True)
    
    with open(dataset_path, "r") as file:
        data = json.load(file)
        print(f"Loaded {len(data)} evaluation questions.")
        
    with open(few_shot_examples_path, "r") as file:
        shot_examples = json.load(file)
        print(f"Loaded {len(shot_examples)} shot examples.")
        
    return {
        "data": data,
        "shot_examples": shot_examples,
        "zero_shot_folder": zero_shot_folder,
        "few_shot_folder": few_shot_folder
    }    
