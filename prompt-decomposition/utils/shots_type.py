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

    # Input Validation
    if few_shot_type not in ["static", "random", "dynamic"]:
        raise ValueError("few_shot_type must be 'static', 'random', or 'dynamic'")
    
    if few_shot_type == "dynamic" and retriever is None:
        raise ValueError("Must provide retriever for dynamic few-shot")

    # Setup 
    system_prompt = model_config.get("system_prompt", "You are an expert assistant that decomposes user questions into a numbered list of simple, sequential sub-questions. Each sub-question should be an answerable query that contributes to answer the initial question. Your output should ONLY be the numbered list of sub-questions. Do not provide the answer or explanations.")
    uses_system_prompt = model_config.get("uses_system_prompt", True)
    generation_params = model_config.get("generation_params", {}).copy()

    # Remove unsupported parameters for certain models
    for unsupported_param in ["temperature", "top_p"]:
        generation_params.pop(unsupported_param, None)


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





import torch
import re
import random

def run_llada_experiment(
    generate_func,
    model,
    tokenizer,
    data,
    shot_examples,
    model_config,
    num_shots,
    few_shot_type="static",   # "static" | "random" | "dynamic"
    retriever=None,           # required if few_shot_type="dynamic"
    seed=42
):
    """
    LLaDA-compatible experiment runner with support for static, random, and dynamic few-shot prompting.
    """

    # Safety checks
    if num_shots > len(shot_examples):
        raise ValueError(f"Requested {num_shots} shots, but only {len(shot_examples)} available.")

    if few_shot_type not in ["static", "random", "dynamic"]:
        raise ValueError("few_shot_type must be 'static', 'random', or 'dynamic'")

    if few_shot_type == "dynamic" and retriever is None:
        raise ValueError("Retriever must be provided for dynamic few-shot mode.")

    # Config
    system_prompt = model_config.get(
        "system_prompt",
        "Decompose the question into sub-questions. Output ONLY numbered sub-questions."
    )
    generation_params = model_config.get("generation_params", {}).copy()

    # LLaDA deterministic settings
    generation_params.update({
        "temperature": 0.0,
        "cfg_scale": 10.0,
        "remasking": "low_confidence"
    })

    results = []

    for item in data:
        try:
            # Few-shot selection
            current_shots = shot_examples
            if few_shot_type == "random":
                random.seed(seed)
                current_shots = random.sample(current_shots, num_shots)
            elif few_shot_type == "dynamic":
                current_shots = retriever.retrieve(item["question"], num_shots)

            # Prompt construction
            prompt_parts = [f"System: {system_prompt}"]

            for example in current_shots[:num_shots]:
                q = example.get("question", "").strip()
                a = example.get("decomposition", [])
                if isinstance(a, list):
                    a = "\n".join(a)
                a_numbered = "\n".join(f"{i+1}. {line.strip()}" for i, line in enumerate(a.split("\n")) if line.strip())

                prompt_parts.append(f"User: {q}")
                prompt_parts.append(f"Assistant: {a_numbered}")

            final_q = item.get("question", "").strip()
            prompt_parts.append(f"User: {final_q}")
            prompt_parts.append("Assistant: 1.")  # start numbering

            full_prompt = "\n\n".join(prompt_parts)

            # Generation
            input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)

            output_ids = generate_func(
                model=model,
                prompt=input_ids,
                **generation_params
            )

            # Only keep new tokens
            response_ids = output_ids[:, input_ids.shape[1]:]
            raw_output = tokenizer.decode(response_ids[0], skip_special_tokens=True)

            # Clean output
            decomposition = []
            for line in raw_output.split("\n"):
                line = line.strip()
                if re.match(r"^\d+\.\s*", line):
                    question = re.sub(r"^\d+\.\s*", "", line).strip()
                    if question and not question.startswith(("System:", "User:", "Assistant:", "<|")):
                        decomposition.append(question)

            # Fallback if no proper numbering
            if not decomposition:
                decomposition = [seg.strip() for seg in raw_output.split("?") if seg.strip()]

        except Exception as e:
            print(f"Error processing ID {item.get('id', 'N/A')}: {str(e)}")
            decomposition = [f"[ERROR: {str(e)}]"]

        # Store results
        results.append({
            "id": item.get("id", "N/A"),
            "original_question": final_q,
            "gold_decomposition": item.get("decomposition", []),
            f"{num_shots}_shot_decomposition": decomposition
        })

    return results
