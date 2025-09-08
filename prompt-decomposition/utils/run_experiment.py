# shots_type.py
import torch
from generate import generate
import re
import random
from retriever import DynamicRetriever
from math import ceil
from tqdm import tqdm
import sys 

def run_experiment(
    model, 
    tokenizer, 
    data, 
    shot_examples, 
    model_config, 
    num_shots,
    few_shot_type="static",  # "static" | "random" | "dynamic"
    retriever=None,          # Required if few_shot_type="dynamic"
    seed=42,
    batch_size=32
):
    """
    Enhanced with all few-shot types.
    
    Args:
        few_shot_type: "static" (default), "random", or "dynamic"
        retriever: Pre-built DynamicRetriever instance (for "dynamic")
        seed: Only used for "random" mode
    """
    if num_shots > len(shot_examples):
        raise ValueError(f"Requested {num_shots} shots, but only {len(shot_examples)} are available.")

    # Input Validation
    if few_shot_type not in ["static", "random", "dynamic"]:
        raise ValueError("few_shot_type must be 'static', 'random', or 'dynamic'")
    
    if few_shot_type == "dynamic" and retriever is None:
        raise ValueError("Must provide retriever for dynamic few-shot")

    # Setup 
    system_prompt = model_config.get("system_prompt")
    uses_system_prompt = model_config.get("uses_system_prompt", True)
    generation_params = model_config.get("generation_params", {}).copy()

    # Remove unsupported parameters for certain models
    for unsupported_param in ["temperature", "top_p"]:
        generation_params.pop(unsupported_param, None)


    # Set pad_token_id if it's not already set. 
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Add essential generation parameters
    generation_params["pad_token_id"] = tokenizer.pad_token_id
    generation_params["eos_token_id"] = tokenizer.eos_token_id

    # Main Loop 
    conversations = []
    for item in data:
        # Dynamic Shot Selection
        if few_shot_type == "random":
            random.seed(seed)
            current_shots = random.sample(shot_examples, num_shots)

        elif few_shot_type == "dynamic":
            current_shots = retriever.retrieve(item["question"], num_shots)
        
        else:
            current_shots = shot_examples[:num_shots]

        # The actual question for the model to answer
        final_user_prompt = item.get('question', '')

        messages = []
        if uses_system_prompt:
            messages.append({"role": "system", "content": system_prompt})                  

            for example in current_shots[:num_shots]:
                q = example.get('question', '')
                a = example.get('decomposition', '')
                if isinstance(a, list):
                    a = "\n".join(a)
                
                messages.append({"role": "user", "content": q})
                messages.append({"role": "assistant", "content": a})
        elif num_shots == 0 and not uses_system_prompt:
            final_user_prompt = f"{system_prompt}\n\n{final_user_prompt}"
        else:
            first_example = current_shots[0]
            q, a = first_example.get("question"), first_example.get("decomposition")
            if isinstance(a, list):
                a = "\n".join(a)

            messages.append({"role": "user", "content": f"{system_prompt}\n\n{q}"})
            messages.append({"role": "assistant", "content": a})

            for example in current_shots[1:num_shots]:
                q = example.get('question', '')
                a = example.get('decomposition', '')
                if isinstance(a, list):
                    a = "\n".join(a)

                messages.append({"role": "user", "content": q})
                messages.append({"role": "assistant", "content": a})        
        
        messages.append({"role": "user", "content": final_user_prompt})
         
        conversations.append((item, messages))
    
    results = []
    num_batches = ceil(len(conversations) / batch_size)
    for batch_idx in tqdm(range(num_batches), desc='Batches'):

        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size
        batch_items= conversations[batch_start:batch_end]
        
        items, messages_batch = zip(*batch_items)


        # GENERATION 
        input_ids = tokenizer.apply_chat_template(
            messages_batch, add_generation_prompt=True, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        
        generation_params["attention_mask"] = (input_ids != tokenizer.pad_token_id).long()
        with torch.no_grad():
            outputs = model.generate(input_ids, **generation_params)
        
        for i, (item, messages) in enumerate(batch_items):
            response_ids = outputs[i, input_ids.shape[-1]:]
            decomposition = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    
            # Store conversations
            results.append({
                "id": item.get('id', 'N/A'),
                "original_question": item.get('question', ''),
                "gold_decomposition": item.get('decomposition', ''),
                f"{num_shots}_shot_decomposition": decomposition
            })

    return results

