# manage_folders.py
import os
import json
import random

def save_results_to_json(results, folder, filename):
    """
    Save data as a JSON file.

    Args:
        results (Any): A JSON-serializable object (e.g., list, dict).
        folder (str): Directory where the file will be saved.
        filename (str): Name of the JSON file.

    Creates the target folder if it does not exist and writes the data
    with UTF-8 encoding.
    """
    os.makedirs(folder, exist_ok=True)
    full_path = os.path.join(folder, filename)
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved successfully to '{full_path}'")


def dataset_folders(
    base_results_folder,
    dataset_path,
    few_shot_examples_path,
    few_shot_type="static",
    tuning_subset_size=500
):
    """
    Prepare dataset splits and results folders for evaluation.

    Args:
        base_results_folder (str): Base directory where results will be stored.
        dataset_path (str): Path to the evaluation dataset (JSON file).
        few_shot_examples_path (str): Path to few-shot examples (JSON file).
        few_shot_type (str, optional): Type of few-shot setup ("static" or other).
            - If "static", both zero-shot and few-shot folders are created.
            - Otherwise, only few-shot is created.
        tuning_subset_size (int, optional): Number of examples to keep for tuning.
            Defaults to 500. Must not exceed dataset size.

    Returns:
        dict: A dictionary containing:
            - "data_full": Full evaluation dataset (list of dicts).
            - "data_subset": First `tuning_subset_size` examples from the dataset.
            - "shot_examples": Loaded few-shot examples.
            - "zero_shot_folder": Path to zero-shot results folder (or None).
            - "few_shot_folder": Path to few-shot results folder.
    """
    # Create results folders
    zero_shot_folder = None
    if few_shot_type == "static":  # Only create for static runs
        zero_shot_folder = os.path.join(base_results_folder, 'zero_shot')
        os.makedirs(zero_shot_folder, exist_ok=True)
    
    few_shot_folder = os.path.join(base_results_folder, 'few_shot')
    os.makedirs(few_shot_folder, exist_ok=True)
    
    # Load dev/eval data
    with open(dataset_path, "r") as file:
        data = json.load(file)
        print(f"Loaded {len(data)} evaluation questions.")
        
    # Load few-shot examples
    with open(few_shot_examples_path, "r") as file:
        shot_examples = json.load(file)
        print(f"Loaded {len(shot_examples)} shot examples.")
        
    # Create tuning subset
    if tuning_subset_size > len(data):
        raise ValueError(f"Tuning subset size ({tuning_subset_size}) is larger than dataset ({len(data)}).")
    dev_subset = data[:tuning_subset_size]
    print(f"Selected {len(dev_subset)} examples from dev set.")

    return {
        "data_full": data,              # Full dev set
        "data_subset": dev_subset,      # Random subset for tuning
        "shot_examples": shot_examples,
        "zero_shot_folder": zero_shot_folder,
        "few_shot_folder": few_shot_folder
    }


