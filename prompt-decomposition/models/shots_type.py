# shots_type.py

import torch
from generate import generate
import re
import random
from retriever import DynamicRetriever



def run_experiment(
    model, 
    tokenizer, 
    data, 
    shot_examples, 
    model_config, 
    num_shots,
    few_shot_type="static",  # "static" | "random" | "dynamic"
    retriever=None,          # Required if few_shot_type="dynamic"
    seed=42
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

    # --- Input Validation ---
    if few_shot_type not in ["static", "random", "dynamic"]:
        raise ValueError("few_shot_type must be 'static', 'random', or 'dynamic'")
    
    if few_shot_type == "dynamic" and retriever is None:
        raise ValueError("Must provide retriever for dynamic few-shot")

    # Setup 
    system_prompt = model_config.get("system_prompt", "You are an expert assistant that decomposes user questions into a numbered list of simple, sequential sub-questions. Each sub-question should be an answerable query that contributes to answer the initial question."
                                     "Your output should ONLY be the numbered list of sub-questions. Do not provide the answer or explanations.")
    uses_system_prompt = model_config.get("uses_system_prompt", True)
    generation_params = model_config.get("generation_params", {}).copy()

    # Set pad_token_id if it's not already set. This is crucial.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Add essential generation parameters
    generation_params["pad_token_id"] = tokenizer.pad_token_id
    generation_params["eos_token_id"] = tokenizer.eos_token_id

    # Main Loop 
    results = []
    for item in data:
        # Dynamic Shot Selection
        current_shots = shot_examples
        if few_shot_type == "random":
            random.seed(seed)
            current_shots = random.sample(current_shots, num_shots)

        elif few_shot_type == "dynamic":
            current_shots = retriever.retrieve(item["question"], num_shots)

        # Prompt Construction 
        messages = []
        if uses_system_prompt:
            messages.append({"role": "system", "content": system_prompt})        

        for example in current_shots[:num_shots]:
            q = example.get('question', '')
            a = example.get('decomposition', '')
            if isinstance(a, list):
                a = "\n".join(a)
            
            # Consistent formatting for all user prompts in examples
            user_prompt = f"{q}\n"
            messages.append({"role": "user", "content": user_prompt})
            messages.append({"role": "assistant", "content": a})

        # Add the final question with consistent formatting
        final_question_text = item.get('question', '')
        final_user_prompt = f"{final_question_text}\n"

        # For zero-shot on models WITHOUT a system prompt, prepend the instructions
        if num_shots == 0 and not uses_system_prompt:
            final_user_prompt = f"{system_prompt}\n{final_user_prompt}"
        
        messages.append({"role": "user", "content": final_user_prompt})

        # GENERATION 
        try:
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            
            generation_params["attention_mask"] = (input_ids != tokenizer.pad_token_id).long()

            outputs = model.generate(input_ids, **generation_params)
            response_ids = outputs[0][input_ids.shape[-1]:]
            decomposition = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        except Exception as e:
            print(f"  ERROR generating response for ID {item.get('id', 'N/A')}: {e}")
            decomposition = f"[GENERATION ERROR: {e}]"

        # STORE RESULTS
        results.append({
            "id": item.get('id', 'N/A'),
            "original_question": item.get('question', ''),
            "gold_decomposition": item.get('decomposition', ''),
            f"{num_shots}_shot_decomposition": decomposition
        })

    return results

# def run_experiment(model, tokenizer, data, shot_examples, model_config, num_shots, random_shots=False, seed=42):
#     """
#     Runs a universal experiment with unified, robust logic for all models.
#     - Handles model-specific system prompt usage.
#     - Consistently formats all prompts for clarity.
#     - Cleans generation parameters for reliable output.
#     """
#     if num_shots > len(shot_examples):
#         raise ValueError(f"Requested {num_shots} shots, but only {len(shot_examples)} are available.")

#     # --- 1. SETUP & CONFIGURATION ---
    
#     # Get configuration safely with defaults
#     system_prompt = model_config.get("system_prompt", "You are an expert assistant that decomposes complex user questions into a numbered list of simple, sequential sub-questions. Each sub-question should be a direct, answerable query that contributes to a logical plan. Your output should ONLY be the sub-questions. Do not provide any explanations or other answers.")
#     uses_system_prompt = model_config.get("uses_system_prompt", True)
#     generation_params = model_config.get("generation_params", {}).copy() # Use .copy() to avoid modifying the original config

#     # Set pad_token_id if it's not already set. This is crucial.
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token_id = tokenizer.eos_token_id
    
#     # Add essential generation parameters
#     generation_params["pad_token_id"] = tokenizer.pad_token_id
#     generation_params["eos_token_id"] = tokenizer.eos_token_id

#     # --- 2. MAIN EXPERIMENT LOOP ---
#     results = []
#     for item in data:
#         print(f"Processing ({num_shots}-Shot) ID: {item.get('id', 'N/A')}")
#         current_shots = shot_examples
#         if random_shots:
#             random.seed(seed)
#             current_shots = random.sample(
#                 [ex for ex in shot_examples if ex.get('id') != item.get('id')], 
#                 min(num_shots, len(shot_examples)-1))


#         # --- 3. PROMPT CONSTRUCTION (CLEANED LOGIC) ---
#         messages = []

#         # Add system prompt if the model supports it
#         if uses_system_prompt:
#             messages.append({"role": "system", "content": system_prompt})

#         # Add Few-shot examples
#         for example in shot_examples[:num_shots]:
#             q = example.get('question', '')
#             a = example.get('decomposition', '')
#             if isinstance(a, list):
#                 a = "\n".join(a)
            
#             # Consistent formatting for all user prompts in examples
#             user_prompt = f"{q}\n\nSub-questions:"
#             messages.append({"role": "user", "content": user_prompt})
#             messages.append({"role": "assistant", "content": a})

#         # Add the final question with consistent formatting
#         final_question_text = item.get('question', '')
#         final_user_prompt = f"{final_question_text}\n\nSub-questions:"

#         # For zero-shot on models WITHOUT a system prompt, prepend the instructions
#         if num_shots == 0 and not uses_system_prompt:
#             final_user_prompt = f"{system_prompt}\n\n{final_user_prompt}"
        
#         messages.append({"role": "user", "content": final_user_prompt})

#         # --- 4. GENERATION ---
#         try:
#             input_ids = tokenizer.apply_chat_template(
#                 messages, add_generation_prompt=True, return_tensors="pt"
#             ).to(model.device)
            
#             generation_params["attention_mask"] = (input_ids != tokenizer.pad_token_id).long()

#             outputs = model.generate(input_ids, **generation_params)
#             response_ids = outputs[0][input_ids.shape[-1]:]
#             decomposition = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
#         except Exception as e:
#             print(f"  ERROR generating response for ID {item.get('id', 'N/A')}: {e}")
#             decomposition = f"[GENERATION ERROR: {e}]"

#         # --- 5. STORE RESULTS ---
#         results.append({
#             "id": item.get('id', 'N/A'),
#             "original_question": item.get('question', ''),
#             "gold_decomposition": item.get('decomposition', ''),
#             f"{num_shots}_shot_decomposition": decomposition
#         })

#     return results




def run_llada_experiment(generate_func, model, tokenizer, data, shot_examples, model_config, num_shots):
    """
    Fixed version for LLaDA with robust prompt handling and output cleaning.
    """
    if num_shots > len(shot_examples):
        raise ValueError(f"Requested {num_shots} shots, but only {len(shot_examples)} are available.")

    # 1. Configuration
    system_prompt = model_config.get("system_prompt", "Decompose the question into sub-questions. Output ONLY numbered sub-questions.")
    generation_params = model_config.get("generation_params", {}).copy()
    
    # Force deterministic generation
    generation_params.update({
        "temperature": 0.0,
        "cfg_scale": 10.0,
        "remasking": "low_confidence"
    })

    results = []
    for item in data:
        try:
            # 2. Build Clean Prompt
            prompt_parts = []
            
            # System prompt (if supported)
            prompt_parts.append(f"System: {system_prompt}")
            
            # Few-shot examples
            for example in shot_examples[:num_shots]:
                q = example.get('question', '').strip()
                a = "\n".join(example.get('decomposition', [])) if isinstance(example.get('decomposition'), list) else example.get('decomposition', '')
                prompt_parts.append(f"User: {q}")
                prompt_parts.append(f"Assistant: 1. {a.replace('\n', '\n2. ')}")  # Ensure numbered list
            
            # Final question
            final_q = item.get('question', '').strip()
            prompt_parts.append(f"User: {final_q}")
            prompt_parts.append("Assistant: 1.")  # Trigger start of decomposition
            
            full_prompt = "\n\n".join(prompt_parts)
            
            # 3. Generate and Clean Output
            input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)
            
            output_ids = generate_func(
                model=model,
                prompt=input_ids,
                **generation_params
            )
            
            # Extract new tokens only
            response_ids = output_ids[:, input_ids.shape[1]:]
            raw_output = tokenizer.decode(response_ids[0], skip_special_tokens=True)
            
            # 4. Post-Processing
            decomposition = []
            for line in raw_output.split('\n'):
                line = line.strip()
                if line.startswith(('1.', '2.', '3.', '4.', '5.')):  # Capture numbered lines
                    question = re.sub(r'^\d+\.\s*', '', line).strip()
                    if question and not question.startswith(('<|', 'System:', 'User:', 'Assistant:')):
                        decomposition.append(question)
            
            # Fallback if numbering failed
            if not decomposition:
                decomposition = [q.strip() for q in raw_output.split('?') if q.strip()]
            
        except Exception as e:
            print(f"Error processing ID {item.get('id')}: {str(e)}")
            decomposition = [f"[ERROR: {str(e)}]"]
        
        # 5. Store Results
        results.append({
            "id": item.get('id', 'N/A'),
            "original_question": final_q,
            "gold_decomposition": item.get('decomposition', []),
            f"{num_shots}_shot_decomposition": decomposition
        })

    return results


# def run_llada_experiment(generate_func, model, tokenizer, data, shot_examples, model_config, num_shots):
#     """
#     Runs a universal experiment specifically for the LLaDA model,
#     using its custom 'generate' function.
#     """
#     if num_shots > len(shot_examples):
#         raise ValueError(f"Requested {num_shots} shots, but only {len(shot_examples)} are available.")

#     # Get configuration safely with defaults
#     system_prompt = model_config.get("system_prompt", "You are an expert assistant that decomposes complex user questions into a numbered list of simple, sequential sub-questions. Each sub-question should be a direct, answerable query that contributes to a logical plan. Your output should ONLY be the sub-questions. Do not provide any explanations or other answers.")
#     generation_params = model_config.get("generation_params", {}).copy()

#     results = []
#     for item in data:
#         print(f"Processing ({num_shots}-Shot) ID: {item.get('id', 'N/A')}")

#         # --- PROMPT CONSTRUCTION ---
#         messages = []
#         # LLaDA uses a Llama base, so it supports a system prompt
#         messages.append({"role": "system", "content": system_prompt})

#         # Add Few-shot examples
#         for example in shot_examples[:num_shots]:
#             q = example.get('question', '')
#             a = example.get('decomposition', '')
#             if isinstance(a, list):
#                 a = "\n".join(a)
#             user_prompt = f"{q}\n\nSub-questions:"
#             messages.append({"role": "user", "content": user_prompt})
#             messages.append({"role": "assistant", "content": a})

#         # Add the final question
#         final_question_text = item.get('question', '')
#         final_user_prompt = f"{final_question_text}\n\nSub-questions:"
#         messages.append({"role": "user", "content": final_user_prompt})

#         # --- GENERATION (LLaDA specific) ---
#         try:
#             # Format the prompt into a single string, as LLaDA's function expects
#             prompt_text = tokenizer.apply_chat_template(
#                 messages, add_generation_prompt=True, tokenize=False
#             ).replace("<|", "").replace("|>", "")  # Remove HF-specific tokens if LLaDA doesn't use them
#             # Tokenize the final string
#             input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)


#             # Call LLaDA's custom generate function with its specific parameters
#             output_ids = generate_func(
#                 model=model,
#                 prompt=input_ids,
#                 **generation_params
#             )

#             response_ids = output_ids[:, input_ids.shape[1]:]
#             decomposition = tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()
#             # Remove any remaining template artifacts:
#             decomposition = re.sub(r'<\|.*?\|>', '', decomposition)  # Strip <|...|> tags
#             decomposition = re.sub(r'\?+', '?', decomposition)  # Fix repeated "???"


#         except Exception as e:
#             print(f"  ERROR generating response for ID {item.get('id', 'N/A')}: {e}")
#             decomposition = f"[GENERATION ERROR: {e}]"

#         # --- STORE RESULTS ---
#         results.append({
#             "id": item.get('id', 'N/A'),
#             "original_question": item.get('question', ''),
#             "gold_decomposition": item.get('decomposition', ''),
#             f"{num_shots}_shot_decomposition": decomposition
#         })

#     return results



# def run_experiment(model, tokenizer, data, shot_examples, model_config, num_shots):
#     """
#     Runs a universal experiment for a given model, controlled by a configuration dictionary.
#     Handles both zero-shot (num_shots=0) and few-shot prompting.
#     """
#     if num_shots > len(shot_examples):
#         raise ValueError(f"You asked for {num_shots} shots, but only {len(shot_examples)} are available.")

#     results = []
#     for item in data:
#         print(f"Processing ({num_shots}-Shot) ID: {item.get('id', 'N/A')}")
        
#         # Build the messages list based on the model_config 
#         messages = []
#         system_prompt = model_config.get("system_prompt", "You are an expert helpful assistant. Your task is to decompose questions into a numbered list of simple sub-questions that form a plan to find the final answer. Do not answer the sub-questions.")

#         # For models that support a system prompt (Llama, Phi-3, etc.)
#         if model_config.get("uses_system_prompt", False):
#             messages.append({"role": "system", "content": system_prompt})

#         # Handle Few-Shot Examples
#         if num_shots > 0:
#             # For models that don't use a system prompt (Gemma, Mistral),
#             # we merge the instructions with the first user message.
#             if not model_config.get("uses_system_prompt", False):
#                 first_question = f"{system_prompt}\n\n{shot_examples[0].get('question', '')}"
#                 # Handle if decomposition is a list
#                 decomposition_content = shot_examples[0].get('decomposition', '')
#                 if isinstance(decomposition_content, list):
#                     decomposition_content = "\n".join(decomposition_content)
#                 messages.append({"role": "user", "content": first_question})
#                 messages.append({"role": "assistant", "content": decomposition_content})
#                 # Add the rest of the examples
#                 for example in shot_examples[1:num_shots]:
#                     decomposition_content = example.get('decomposition', '')
#                     if isinstance(decomposition_content, list):
#                         decomposition_content = "\n".join(decomposition_content)
#                     messages.append({"role": "user", "content": example.get('question', '')})
#                     messages.append({"role": "assistant", "content": decomposition_content})
#             else:
#                 # For other models, just add the examples normally
#                 for example in shot_examples[:num_shots]:
#                     decomposition_content = example.get('decomposition', '')
#                     if isinstance(decomposition_content, list):
#                         decomposition_content = "\n".join(decomposition_content)
#                     messages.append({"role": "user", "content": example.get('question', '')})
#                     messages.append({"role": "assistant", "content": decomposition_content})

#         # Add the Final User Question
#         final_question = item.get('question', '')
#         # If it's a zero-shot run for a model without a system prompt, add instructions here
#         if num_shots == 0 and not model_config.get("uses_system_prompt", False):
#             final_question = f"{system_prompt}\n\nQuestion\n{final_question}"
        
#         messages.append({"role": "user", "content": final_question})

#         # Generate the Response 
#         input_ids = tokenizer.apply_chat_template(
#             messages, add_generation_prompt=True, return_tensors="pt"
#         ).to(model.device)

#         attention_mask = torch.ones_like(input_ids)

#         # Use model-specific generation parameters from the config
#         generation_params = model_config.get("generation_params", {})
        

#         #  Add attention_mask to the parameters dictionary
#         generation_params['attention_mask'] = attention_mask
        
#         # Use the correct tokenizer attribute: .pad_token_id (the integer), not .pad_token (the string)
#         #  Also, ensure the tokenizer has a pad_token_id set.
#         if tokenizer.pad_token_id is None:
#             tokenizer.pad_token_id = tokenizer.eos_token_id
#         generation_params['pad_token_id'] = tokenizer.pad_token_id
        
#         outputs = model.generate(input_ids, **generation_params)
        
#         response_ids = outputs[0][input_ids.shape[-1]:]
#         decomposition_result = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

#         results.append({
#             "id": item.get('id', 'N/A'),
#             "original_question": item.get('question', ''),
#             "gold_decomposition": item.get('decomposition', 'N/A'),
#             f"{num_shots}_shot_decomposition": decomposition_result
#         })
#     return results
