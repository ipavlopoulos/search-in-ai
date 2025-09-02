### **Comparative Study of Small LLMs for Complex Question Decomposition**

---

#### $Project$ $Overview$

---

This repository supports a comparative study on how various **small Large Language Models (LLMs)** perform **question decomposition**. The main goal is to see how different models break down a complex query into a sequence of simpler, logically ordered sub-questions.

Unlike traditional QA benchmarks, the focus here is **not on finding the final answer**, but on evaluating the **quality of the decomposition** itself.

This repo is intended to:
- Help reproduce or extend the decomposition prompting experiments.
- Serve as a shared codebase for testing different models and prompt strategies.

> **Note:** This is *not* a full project. That will be hosted in a separate repository and will include full analysis, evaluation metrics, and results.

---

#### $Models$ $Tested$

---

We evaluate a range of small and efficient LLMs for few-shot and zero-shot decomposition:

- **LLaMA 3 Family**: `Llama-3.2-3B`, `Llama-3.1-8B`
- **Mistral**: `Mistral-7B-Instruct-v0.2`
- **Gemma Family**: `gemma-2b-it`, `gemma-7b-it`
- **Phi Family**: `Phi-3-mini-4k-instruct`, `Phi-3-small-8k-instruct`
- **TinyLlama**: `TinyLlama-1.1B-Chat-v1.0`

---

#### $Datasets$

---

We evaluate decomposition performance across multiple benchmark datasets:

- **HotpotQA**: Multi-hop reasoning across documents (implicit decomposition)
- **StrategyQA**: Implicit multi-step reasoning for yes/no classification
- **QDMR**: Explicit human-annotated question decompositions

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

To illustrate differences in model behavior, we present decompositions generated for the following question from the **QDMR** dataset predicted with the Dynamic 3-shot learning:

> **Question:** *"What is the most common language in the country where Jamaican English is spoken?"*

#####  Ground Truth Decomposition

1. return the country where Jamaican English is spoken
2. return what is the most common language in #1
   

##### Some Model Outputs 

| **Model**          | **Decomposition Output**                                                                                                                                                                                    |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LLaMA-3.1-8B**    | 1. What is the name of the country where Jamaican English is spoken?  <br> 2. Is Jamaican English the most widely used language there? <br> 3. If not, what is the most commonly spoken language in that country?   |
| **LLaMA-3.2-3B**   | 1. return country where Jamaican English is spoken  <br> 2.return #1 most common language   |
| **Mistral-7B**     | 1. Which country is Jamaican English spoken in? <br> 2. What is the most common language in that country?       |
| **Gemma-2B-it**    | 1. return most common language in country where Jamaican English is spoken <br> 2. return #1 which is Jamaican English      |
| **Gemma-7B-it**    | 1. return the most common language in the country where Jamaican English is spoken <br> 2. return the name of the country where Jamaican English is spoken in <br> 3. Return the number of states in the United States of America. <br> 3. return the name of the state that has the largest population in the United States of America |
| **Phi-3-mini** | 1. return common language in the country where Jamaican English is spoken <br> 2. return #1 English is the most common language in the country where Jamaican English is spoken, which is Jamaica. In Jamaica, English is the official language, but it is spoken differently in various parts of the country. In Urban areas, residents often mix English with Jamaican Patois (a creole language influenced by West African, Irish, and English languages) to create what is often referred to as \"Jamaican English.\" This mixture reflects the multilingual and multicultural context of Jamaica's society. Despite English's role as the official language, Jamaican Patois also has official status and is widely used in local communication. Nonetheless, when considering the language as an independent phenomenon, English remains the most prevalent and primary form of communication in all official capacities in Jamaica.
 |
| **Phi-4-mini** | 1. return most common language in Jamaica <br> 2. return Jamaican English languagereturn country where Jamaican English is spoken <br> 3. return #1 most common language in that countryreturn most common language in Jamaica <br> 4. return Jamaican English is spoken in Jamaica <br> 5. return Jamaica is a country in the Caribbean <br> 6.return Caribbean is a geographical region mostly in the Atlantic Ocean <br> 7. return Atlantic Ocean is a large body of water between the Americas to the east and Europe and Africa to the west. <br> 8. return English is the most commonly spoken language in Jamaica. <br> 9. *Then it returns an answer and notes*   |
| **TinyLlama**    | 1. return most common language in the country where Jamaican English is spoken  <br> 2. return #1 that is spoken in that country   |             



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
│   │   │   └── few_shot/
│   │   └── dynamic/
│   │       └── few_shot/
│   └── data_preprocess/
│       └── HotpotQA_data_preparation.ipynb
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
│   │   │   └── few_shot/
│   │   └── dynamic/
│   │       └── few_shot/
│   └── data_preprocess/
│       └── StrategyQA_data_preparation.ipynb
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
│   │   │   └── few_shot/
│   │   └── dynamic/
│   │       └── few_shot/
│   └── data_preprocess/
│       ├── app_store_generation.py
│       └── QDMR_data_preparation.ipynb
│
├── model_experiments/                     # Prompt decomposition runs by model
│   ├── 01_experiment_Llama-3.2-3B.ipynb
│   ├── 02_experiment_Llama-3.1-8B.ipynb
│   ├── 03_experiment_Mistral-7B.ipynb
│   ├── 04a_experiment_Gemma-2B.ipynb
│   ├── 04b_experiment_Gemma-7B.ipynb
│   ├── 05a_experiment_Phi-3-Mini.ipynb
│   ├── 05b_experiment_Phi-3-Small.ipynb
│   └── 06_experiment_TinyLlama.ipynb           
│       
├── utils/                     # Library with fhelper functions
│   ├── __init__.py
│   ├── generate.py           # LLaDA required file to run from github
│   ├── manage_folders.py     # handles the predictions saving and the datasets loading          
│   ├── retriever.py          # class for the dynamic shot_type
│   └── run_experiment.py     # Handles the num of shots(0 or more), the type of model and the shot_type: random, static, dynamic **    
│
├── requirements.txt          # Libraries Required to install
│
└── README.md


````

---

#### $Current$ $Status$ $and$ $Roadmap$

---

##### Completed

* Baseline runs on **HotpotQA**, **StrategyQA** and **QDMR**
* Prompt templates for all tested models
* Zero-shot and few-shot decomposition 
* Static, Random and Dynamic predictions - 3shot learning 
* JSON-based output storage pipeline

---

#### $More$ $options$ $to$ $explore$ $that$ $haven't$ $made$ $predictions$ $for$

---

* *Change the system prompt*

* *Adjust the order of examples introduced to the model*
  Specify the order strategy in:

  ```python
  retriever_instance = DynamicRetriever(shot_examples, order_strategy)
  ```

  By default, the strategy is `"first"`. Options to explore:

  * `"first"`: most similar examples first
  * `"last"`: most similar examples last
  * `"ushaped"`: place the most similar example in the center; remaining examples are arranged symmetrically around it to form a U-shape (least-to-most and most-to-least similar)

* *Adjust the number of shots or batch size*

  ```python
  run_experiment(
      model=model,
      tokenizer=tokenizer,
      data=qdmr_data,
      shot_examples=shot_examples,
      model_config=model_config,   # Use the same config
      num_shots=3,                 # number of shots
      few_shot_type="random",      # "static" | "random" | "dynamic"
      retriever=None,              # Required if few_shot_type="dynamic"
      seed=42,
      batch_size=5
  )
  ```

* *Try more datasets with additional examples*
  Examples: `$Musique$` or `$GSM8K$` (these have gold decompositions).
  You can adjust the number of examples by changing `tuning_subset_size`:

  ```python
  data_assets = dataset_folders(
      qdmr_base_folder, 
      qdmr_dataset_file, 
      qdmr_fewshot_file, 
      few_shot_type="static", 
      tuning_subset_size=5
  )
  ```

  > Note: In this implementation, the dataset files contain only 5 examples, so experimenting with larger subsets may require adding more examples.


---

#### $Setup$ $and$ $Installation$

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

   * Navigate to `model_experiments/` and open any experiment (e.g., `03_experiment_Mistral-7B.ipynb`).

3. **Run the cells in order:**

   * The notebook will load the model, run decomposition for each sample in the dataset, and save the results.



