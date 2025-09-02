from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoModelForCausalLM

class DynamicRetriever:

    """
    A class to retrieve the most relevant few-shot examples (shots) for a given target question, 
    using embeddings to measure semantic similarity. Supports different ordering strategies 
    including a U-shaped arrangement.

    Attributes:
        shot_examples (list of dict): The list of few-shot examples, each with a 'question' key.
        order_strategy (str): Strategy for ordering retrieved examples. Options:
            - "first": most similar first
            - "last": most similar last
            - "ushaped": most similar example in the center, remaining examples placed symmetrically
                        around it to form a U-shape from least-to-most and most-to-least similar.
        device (str): Device to run embedding computations on, e.g., 'cuda' or 'cpu'.
        embed_model: Model used to compute embeddings: gte-multilingual-base.
        use_gte (bool): gte is used as the embedding model.
        corpus_embeddings (tensor or array): Precomputed embeddings of all shot examples.
    """

    
    def __init__(
        self, 
        shot_examples, 
        order_strategy="first", 
        device='cuda'
    ):

        """
        Initialize the DynamicRetriever.

        Args:
            shot_examples (list of dict): Few-shot examples to be used for retrieval.
            order_strategy (str): Retrieval order strategy. Defaults to "first".
            device (str): Device to run embeddings on. Defaults to 'cuda'.
        """
        self.shot_examples = shot_examples
        self.order_strategy = order_strategy
        self.device = device

        print("Using gte-multilingual-base for embeddings...")
        self.embed_model =  SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True, device=device)
        self.use_gte = True

        # Precompute embeddings
        corpus_questions = [ex.get('question', '') for ex in shot_examples]
        self.corpus_embeddings = self.embed_model.encode(
            corpus_questions, convert_to_tensor=True, device=device
        )


    def retrieve(self, target_question, num_shots):

        """
        Retrieve the most relevant few-shot examples for a given target question.

        Args:
            target_question (str): The question for which to retrieve examples.
            num_shots (int): Number of examples to retrieve.

        Returns:
            list of dict: Retrieved examples ordered according to `order_strategy`.
        """

         # Compute query embedding
        query_embedding = self.embed_model.encode(target_question, convert_to_tensor=True, device=self.device)
        top_k_hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=num_shots)[0]

        if self.order_strategy == "first":
            retrieved_examples = [self.shot_examples[hit['corpus_id']] for hit in top_k_hits]

        elif self.order_strategy == "last":
            retrieved_examples = [self.shot_examples[hit['corpus_id']] for hit in reversed(top_k_hits)]

        elif self.order_strategy == "ushaped":
            sorted_examples = [self.shot_examples[hit['corpus_id']] for hit in top_k_hits]
            center = sorted_examples[0]  # most similar in the middle
            remaining = sorted_examples[1:]  # all others

            left, right = [], []
            for i, ex in enumerate(remaining):
                if i % 2 == 0:
                    left.append(ex)
                else:
                    right.append(ex)

            retrieved_examples = list(reversed(left)) + [center] + right
            retrieved_examples = retrieved_examples[:num_shots]  # trim if needed

        else:
            raise ValueError(f"Unknown order_strategy: {self.order_strategy}")

        return retrieved_examples
    



