# retriever.py file

from sentence_transformers import SentenceTransformer, util
import torch

class DynamicRetriever:
    """
    A class to handle dynamic few-shot example retrieval based on semantic similarity.
    """
    def __init__(self, shot_examples, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initializes the retriever by pre-computing embeddings for all shot examples.
        
        Args:
            shot_examples (list): A list of dictionaries, where each dict has 'question' and 'decomposition'.
            device (str): The device to run the embedding model on ('cuda' or 'cpu').
        """
        print("Initializing DynamicRetriever...")
        # A good, lightweight model for calculating sentence similarity
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.shot_examples = shot_examples
        
        # Pre-compute embeddings for all example questions for fast retrieval later
        corpus_questions = [example.get('question', '') for example in self.shot_examples]
        self.corpus_embeddings = self.model.encode(corpus_questions, convert_to_tensor=True, device=device)
        print("Retriever initialized and example embeddings are pre-computed.")

    def retrieve(self, target_question, num_shots):
        """
        Finds the N most semantically similar examples for a given target question.
        
        Args:
            target_question (str): The question you want to find examples for.
            num_shots (int): The number of similar examples to retrieve.
            
        Returns:
            list: A list of the most similar shot example dictionaries.
        """
        # Encode the target question to get its embedding
        query_embedding = self.model.encode(target_question, convert_to_tensor=True, device=self.model.device)
        
        # Perform a semantic search to find the top_k most similar examples
        top_k_hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=num_shots)[0]
        
        # Get the original example dictionaries for the top results
        retrieved_examples = [self.shot_examples[hit['corpus_id']] for hit in top_k_hits]
        
        return retrieved_examples