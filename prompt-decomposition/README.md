### **Comparative Study of Small LLMs for Complex Question Decomposition**

---

#### $Project$ $Overview$

---

This repository supports a comparative study on how various **small Large Language Models (LLMs)** perform **complex question decomposition**. The main goal is to test how well these models break down a complex query into a sequence of simpler, logically ordered sub-questions.

Unlike traditional QA benchmarks, the focus here is **not on finding the final answer**, but on evaluating the **quality of the decomposition** itself.

This repo is intended to:
- Help reproduce or extend the decomposition prompting experiments.
- Serve as a shared codebase for testing different models and prompt strategies.

> **Note:** This is *not* a full capstone research project. That will be hosted in a separate repository and will include full analysis, evaluation metrics, and results.

---

#### $Models$ $Tested$

---

We evaluate a range of small and efficient LLMs for few-shot and zero-shot decomposition:

- **LLaMA 3 Family**: `Llama-3.2-3B`, `Llama-3.1-8B`
- **Mistral**: `Mistral-7B-Instruct-v0.2`
- **Gemma Family**: `gemma-2b-it`, `gemma-7b-it`
- **Phi Family**: `Phi-3-mini-4k-instruct`, `Phi-3-small-8k-instruct`
- **TinyLlama**: `TinyLlama-1.1B-Chat-v1.0`
- **LLaDA**: `LLaDA-8B-Base`
- **BitNet**: `bitnet-b1.58-2B-4T` *(planned)*

---

#### $Datasets$

---

We evaluate decomposition performance across multiple benchmark datasets:

- **HotpotQA**: Multi-hop reasoning across documents (implicit decomposition)
- **StrategyQA**: Implicit multi-step reasoning for yes/no classification
- **QDMR**: Explicit human-annotated question decompositions

> Future work will include: gsm8k (contains both decomposition steps and answers)

---

#### $Methodology$

---

##### Prompting Strategies

Each model is tested with the following strategies:

- **Zero-Shot Prompting**: Only a system prompt describing the task is provided.
- **Few-Shot Prompting**: A system prompt and 3 high-quality examples are given in conversational format (model-specific chat templates).
- **Static Prompting**: Predefined examples selected manually.
- **Dynamic Prompting**: Shot selection based on question content 
- **Random Prompting**: Shot selection based on randomness.

Decomposition outputs are saved as structured `.json` files for each strategy in the `llm_predictions/` directories.

##### Model Loading

- Models are loaded using Hugging Face’s `transformers` library.
- 4-bit quantization (`bitsandbytes`) is supported for efficient inference.

---

#### $Results$

---


##### Sample Decomposition Outputs (Few-Shot Prompting)

To illustrate differences in model behavior, we present decompositions generated for the following question from the **QDMR** dataset:

> **Question:** *"What is the most common language in the country where Jamaican English is spoken?"*

#####  Ground Truth Decomposition

1. return the country where Jamaican English is spoken
2. return what is the most common language in #1



##### Some Model Outputs 

| **Model**          | **Decomposition Output**                                                                                                                                                                                    |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LLaMA-3.2-3B**   | 1. return the most common language in the country where Jamaican English is spoken  <br> 2. return the name of the country where Jamaican English is spoken. <br> *Then the model set another question and left "======" for answer*            |
| **Mistral-7B**     | 1. return most common language in country where Jamaican English is spoken  <br> 2. English (Jamaican Patois or Creole is widely used as an informal dialect but English remains the official language) <br> *Includes verbose explanation; not decomposed into steps*       |
| **TinyLlama-1.1B** | 1. return the most common language in the country where Jamaican English is spoken  <br> 2. return #1 that is spoken in the country where Jamaican English is spoke                                                                            |
| **Gemma-2B-it**    | 1. return most common language in country where Jamaican English is spoken <br> 2. return Jamaica            |
| **Gemma-7B-it**    | 1. Return  the answer to this question:  <br> 2. The majority languages are Hindi-Urdu (official) with more than half its speakers living outside India; other major Indian Languages include Bengali , Telugu & Marathi .                                                                         |




---

#### $Project$ $Structure$

---

```markdown
prompt-decomposition/
├── HotpotQA/
│   ├── HotpotQA_examples/
│   │   ├── hotpot_evaluation.json
│   │   └── hotpot_few_shot.json
│   ├── llm_predictions/
│   │   ├── static/
│   │   │   ├── zero_shot/
│   │   │   └── few_shot/
│   │   ├── random/
│   │   │   └── zero_shot/
│   │   └── dynamic/
│   │       └── zero_shot/
│   └── notebooks/
│       ├── HotpotQA_data_preparation.ipynb
│       └── HotpotQA_results.ipynb           # Will have the results printed and comparison through the models of this dataset
│
├── StrategyQA/
│   ├── StrategyQA_examples/
│   │   ├── strategyqa_evaluation.json
│   │   └── strategyqa_few_shot.json
│   ├── llm_predictions/
│   │   ├── static/
│   │   │   ├── zero_shot/
│   │   │   └── few_shot/
│   │   ├── random/
│   │   │   └── zero_shot/
│   │   └── dynamic/
│   │       └── zero_shot/
│   └── notebooks/
│       ├── StrategyQA_data_preparation.ipynb
│       └── StrategyQA_results.ipynb
│
├── QDMR/
│   ├── QDMR_examples/
│   │   ├── qdmr_evaluation.json
│   │   └── qdmr_few_shot.json
│   ├── llm_predictions/
│   │   ├── static/
│   │   │   ├── zero_shot/
│   │   │   └── few_shot/
│   │   ├── random/
│   │   │   └── zero_shot/
│   │   └── dynamic/
│   │       └── zero_shot/
│   └── notebooks/
│       ├── app_store_generation.py
│       ├── QDMR_data_preparation.ipynb
│       └── QDMR_results.ipynb
│
├── models/                     # Prompt decomposition runs by model
│   ├── 01_experiment_Llama-3.2-3B.ipynb
│   ├── 02_experiment_Llama-3.1-8B.ipynb
│   ├── 03_experiment_Mistral-7B.ipynb
│   ├── 04a_experiment_Gemma-2B.ipynb
│   ├── 04b_experiment_Gemma-7B.ipynb
│   ├── 05a_experiment_Phi-3-Mini.ipynb
│   ├── 05b_experiment_Phi-3-Small.ipynb
│   ├── 06_experiment_TinyLlama.ipynb
│   ├── 07_experiment_LLaDA.ipynb
│   ├── manage_folders.py              # handles the predictions saving and the datasets loading
│   ├── shots_type.py                  # Handles the num of shots(0 or more), the type of model and the shot_type: random, static, dynamic **
│   ├── retriever.py                   # class for the dynamic shot_type
│   └── generate.py                    # LLaDA required file to run from github 
│       # BitNet notebook to be added
│
└── README.md


** There is a run_llada_experiment function as well in this .py file but is not ready yet.
````

---

#### $Current$ $Status$ $&$ $Roadmap$

---

##### Completed

* Baseline runs on **HotpotQA**, **StrategyQA** and **QDMR**
* Prompt templates for all tested models
* Zero-shot and few-shot decomposition prompts
* JSON-based output storage pipeline

##### Next Steps

1. **Dataset Expansion** – Run experiments for gsm8k
2. **Different Number of Shots** - Experiment and find the ideal number for each model and type of question 
3. **Decomposition Correlation with Accuracy** - use gsm8k that contains both answers and decomposition
---

#### $Setup$ $&$ $Installation$

---

**Requirements**

> Python 3.9+ and a CUDA-enabled GPU with at least 8GB VRAM is recommended.

```bash
# Clone the repository
git clone https://github.com/LefkiAth/prompt-decomposition.git
cd prompt-decomposition

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

#### $Running$ $the$ $Experiments$

---

1. **Authenticate (if needed):**

   * For gated models like LLaMA or Gemma, run `notebook_login()` in the notebook and enter your Hugging Face token.

2. **Choose a model notebook:**

   * Navigate to `models/` and open any experiment (e.g., `03_experiment_Mistral-7B.ipynb`).

3. **Run the cells in order:**

   * The notebook will load the model, run decomposition for each sample in the dataset, and save the results.



