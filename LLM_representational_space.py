import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Optional
from tqdm import tqdm

# For embeddings
import openai
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


class MDSProjector:
    """
    A class that orchestrates sentence generation, embedding retrieval, MDS projection, and result plotting.
    Designed for both numeric (1–10) and categorical (bad, not good, ...) rating scales.
    
    Example Usage:
    --------------
    >>> concepts = ["judging beauty", "rating spiciness"]
    >>> mds_proj = MDSProjector(api_key="YOUR_OPENAI_API_KEY")
    >>> sentences = mds_proj.generate_sentences(concepts, categorical=False)
    >>> embeddings = mds_proj.get_embeddings(sentences, model_name="text-embedding-3-small")
    >>> points_2d = mds_proj.apply_mds(embeddings)
    >>> mds_proj.plot_2d(points_2d, labels=[s['score'] for s in sentences])
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the MDSProjector.

        Parameters
        ----------
        api_key : str, optional
            API key for services like OpenAI, Voyage, etc.
        """
        self.api_key = api_key  # Store for use in embedding methods
        self.scores = list(range(1, 11))  # Default numerical scores
    
    def generate_sentences(
        self, 
        concepts: List[str], 
        categorical: bool = False, 
        num_phrasings: int = 10
    ) -> List[dict]:
        """
        Generates multiple sentences for each concept with phrasing variations.
        Supports both categorical and numerical scoring systems.
    
        Parameters
        ----------
        concepts : List[str]
            List of concept topics.
        categorical : bool, default=False
            If True, uses categorical labels instead of numerical scores.
        num_phrasings : int, default=10
            Number of phrasing variations to include.
    
        Returns
        -------
        List[dict]
            Each dictionary contains 'concept', 'score', 'text', and 'phrasing'.
        """
        if categorical:
            self.scores = ["abysmal", "terrible", "dreadful", "inadequate", "mediocre", "decent", "acceptable",  "good", "great", "excellent",
                           "superb", "magnificent", "perfect"]
            self.scores = [ "F", "D−", "D", "D+","C−", "C", "C+", "B−", "B", "B+","A−","A","A+"]
            # self.scores = [f'{i}%' for i in range(10,101,10)]
            
            phrasings = [
                lambda concept, score: f"We are {concept}, my evaluation is {score}",
                lambda concept, score: f"Rated as {score} after {concept}",
                lambda concept, score: f"After {concept}, receiving a {score} rating",
                lambda concept, score: f"On the topic of {concept}, my score is {score}",
                lambda concept, score: f"{concept.capitalize()} gets a rating of {score}",
                lambda concept, score: f"My {concept} judgement is {score}",
                lambda concept, score: f"{concept.capitalize()} is evaluated as {score}",
                lambda concept, score: f"The rating for {concept} is {score}",
                lambda concept, score: f"{concept.capitalize()} scores as {score}",
                lambda concept, score: f"In terms of {concept}, my score is {score}",
            ][:num_phrasings]
        else:
            
            # Or with list comprehension
            self.scores = range(1, 11,1)
            max_score = max(self.scores)
            phrasings = [
                lambda concept, score: f"We are {concept} on a scale of 1 to {max_score}, my score is {score}",
                lambda concept, score: f"Rated with {score}/{max_score} after {concept}",
                lambda concept, score: f"After {concept}, receiving {score} out of {max_score}",
                lambda concept, score: f"On the topic of {concept}, my score is {score} out of {max_score}",
                lambda concept, score: f"{concept.capitalize()} gets a rating of {score} out of {max_score}",
                lambda concept, score: f"My {concept} score is {score} out of {max_score}",
                lambda concept, score: f"{concept.capitalize()} is evaluated as {score}/{max_score}",
                lambda concept, score: f"The rating for {concept} is {score} on a scale of 1 to {max_score}",
                lambda concept, score: f"{concept.capitalize()} scores a {score} out of {max_score}",
                lambda concept, score: f"In terms of {concept}, my score is {score}/{max_score}",
            ][:num_phrasings]
        
        all_sentences = []
        for concept in concepts:
            for score in self.scores:
                for phrasing_index, phrasing in enumerate(phrasings, start=1):
                    txt = phrasing(concept, score)
                    all_sentences.append({
                        'concept': concept,
                        'score': score,
                        'text': txt,
                        'phrasing': phrasing_index
                    })
        return all_sentences


    def get_embeddings(self, sentences: List[Union[str, dict]], 
                       model_name: str,
                      OPENAI_APIKEY: str) -> np.ndarray:
        """
        Obtain embeddings for a list of sentences from various models (OpenAI, HF, Voyage, etc.).

        Parameters
        ----------
        sentences : list of (str or dict)
            If dict, must contain 'text' key. Otherwise, each is a raw string.
        model_name : str
            Identifier for the embedding model (e.g. "text-embedding-3-small", "sentence-transformers/all-MiniLM-L6-v2", etc.).
        OPENAI_APIKEY: str
            api key for openai
        
        Returns
        -------
        embeddings : np.ndarray of shape (N, D)
            Where N is the number of sentences, and D is the embedding dimension.
        """
        # Extract plain strings if needed
        texts = [s['text'] if isinstance(s, dict) else s for s in sentences]

        # Route to the correct method
        if model_name in ("text-embedding-3-large", "text-embedding-3-small"):
            return self._get_openai_embeddings(texts, model_name, OPENAI_APIKEY)
        elif model_name.startswith("voyage"):
            return self._get_voyage_embeddings(texts, model_name)
        elif model_name.startswith("claude"):
            return self._get_claude_embeddings(texts, model_name)
        else:
            return self._get_hf_embeddings(texts, model_name)

    def _get_openai_embeddings(self, texts: List[str], model_name: str, OPENAI_APIKEY: str ) -> np.ndarray:
        """
        Retrieve embeddings using the OpenAI API.
        """
        if not self.api_key:
            raise ValueError("OpenAI API key is not provided.")

        openai.api_key = self.api_key
        embeddings = []
        from tqdm import tqdm  # local import to avoid polluting top-level
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", OPENAI_APIKEY))
        
        for txt in tqdm(texts, desc=f"Fetching OpenAI ({model_name}) embeddings"):
            try:
                emb = client.embeddings.create(
                    input=txt, 
                    model=model_name
                ).data[0].embedding
                embeddings.append(emb)
            except Exception as e:
                print(f"Error for '{txt[:50]}...': {e}")
                # Use zero vector matching model dimensions
                dim = 3072 if model_name == "text-embedding-3-large" else 1536
                embeddings.append([0.0] * dim)

        return np.array(embeddings, dtype=np.float32)

    def _get_voyage_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """
        Retrieve embeddings from Voyage (pseudo-code).
        """
        # import voyageai  # would require an actual import
        if not self.api_key:
            raise ValueError("Voyage API key is not provided.")

        print(f"Mock embedding call for Voyage model: {model_name}")
        # Pseudo-code if voyageai.Client is used
        # vo = voyageai.Client(api_key=self.api_key)
        # result = vo.embed(texts, model=model_name, input_type="document")
        # return np.array(result.embeddings, dtype=np.float32)

        # For demonstration, return random vectors:
        return np.random.rand(len(texts), 1024).astype(np.float32)

    def _get_claude_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """
        Retrieve embeddings from Claude (pseudo-code).
        """
        if not self.api_key:
            raise ValueError("Anthropic/Claude API key is not provided.")

        print(f"Mock embedding call for Claude model: {model_name}")
        # Actual implementation would use Anthropic's library or endpoints
        # For demonstration, return random vectors:
        return np.random.rand(len(texts), 768).astype(np.float32)

    def _get_hf_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """
        Retrieve embeddings from a Hugging Face transformer model.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Error loading HF model {model_name}: {e}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()

        embeddings = []
        from tqdm import tqdm  # local import
        for txt in tqdm(texts, desc=f"Fetching HF ({model_name}) embeddings"):
            try:
                inputs = tokenizer(txt, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                # Mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            except Exception as e:
                print(f"Error for '{txt[:50]}...': {e}")
                embedding = np.zeros(model.config.hidden_size, dtype=np.float32)
            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)

    @staticmethod
    def save_embeddings(embeddings: np.ndarray, filepath: str = "embeddings.pkl") -> None:
        """
        Saves embeddings to a local file.
        """
        with open(filepath, "wb") as f:
            pickle.dump(embeddings, f)

    @staticmethod
    def load_embeddings(filepath: str = "embeddings.pkl") -> np.ndarray:
        """
        Loads embeddings from a local file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Embedding file not found: {filepath}")

        with open(filepath, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def apply_mds(
        embeddings: np.ndarray,
        scale: bool = True,
        n_init: int = 10,
        max_iter: int = 1500,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Applies MDS to reduce embeddings to 2D.

        Parameters
        ----------
        embeddings : np.ndarray of shape (N, D)
            N data points, each D-dimensional.
        scale : bool, default=True
            Whether to standardize embeddings before MDS.
        n_init : int, default=10
            Number of times the MDS algorithm will be run with different initial solutions.
        max_iter : int, default=1500
            Maximum number of iterations of the MDS algorithm.
        random_state : int, default=42
            Determines random number generation for MDS.

        Returns
        -------
        points_2d : np.ndarray of shape (N, 2)
            2D projection of the input embeddings via MDS.
        """
        # Optionally scale
        if scale:
            scaler = StandardScaler()
            embeddings = scaler.fit_transform(embeddings)

        # Convert to cosine dissimilarity
        sim_matrix = cosine_similarity(embeddings)
        diss_matrix = 1.0 - sim_matrix

        # Perform MDS
        mds = MDS(
            n_components=2,
            metric=True,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            dissimilarity='precomputed'
        )
        points_2d = mds.fit_transform(diss_matrix)
        return points_2d

    def compute_mds_projections(
        self,
        embeddings: np.ndarray,
        num_topics: int,
        num_ratings: int,
        num_phrasings: int
    ) -> tuple:
        """
        Computes MDS projections at the topic-rating-phrasing level, as well as an MDS on the
        averaged dissimilarity matrix. Additionally, aligns each 2D projection to the first
        using a Procrustes transformation to maintain a consistent orientation.
        """
        from tqdm import tqdm
        from scipy.spatial import procrustes
    
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D: (n_samples, embedding_dim).")
    
        embedding_dim = embeddings.shape[1]
        expected_size = num_topics * num_ratings * num_phrasings
        if embeddings.shape[0] != expected_size:
            raise ValueError(
                f"Expected {expected_size} embeddings, got {embeddings.shape[0]}."
            )
    
        # Reshape to: (num_topics, num_ratings, num_phrasings, embedding_dim)
        reshaped = embeddings.reshape(num_topics, num_ratings, num_phrasings, embedding_dim)
        # Reorder to group by (topic, phrasing) => shape = (num_phrasings*num_topics, num_ratings, embedding_dim)
        reordered = reshaped.transpose(2, 0, 1, 3).reshape(num_phrasings * num_topics, num_ratings, embedding_dim)
    
        # Prepare MDS
        mds = MDS(
            n_components=2,
            metric=True,
            n_init=10,
            max_iter=1500,
            random_state=42,
            dissimilarity='precomputed'
        )
    
        all_topics_points_2d = []
        all_dissimilarity_matrices = []
    
        # Reference for alignment
        reference_2d = None
    
        # Each 'block' => shape (num_ratings, embedding_dim)
        for block in tqdm(reordered, desc="Computing MDS for each block"):
            sim_matrix = cosine_similarity(block)
            diss_matrix = 1.0 - sim_matrix
            np.fill_diagonal(diss_matrix, 0.0)
            all_dissimilarity_matrices.append(diss_matrix)
    
            points_2d = mds.fit_transform(diss_matrix)  # (num_ratings, 2)
    
            # Align each new 2D projection to the first using Procrustes for consistent orientation
            if reference_2d is None:
                reference_2d = points_2d.copy()
            else:
                _, aligned_points, _ = procrustes(reference_2d, points_2d)
                points_2d = aligned_points
    
            all_topics_points_2d.append(points_2d)
    
        avg_dissimilarity = np.mean(all_dissimilarity_matrices, axis=0)  # (num_ratings, num_ratings)
        avg_points = mds.fit_transform(avg_dissimilarity)
    
        # Optionally align the average points as well
        _, avg_points_aligned, _ = procrustes(reference_2d, avg_points)
        avg_points = avg_points_aligned
    
        return all_topics_points_2d, avg_points, all_dissimilarity_matrices
    

    
    def plot_results(
        self,
        all_topics_points_2d: List[np.ndarray],
        avg_points: np.ndarray,
        num_ratings: int,
        selected_phrasings: Optional[List[int]] = None,
        categorical: bool = False
    ) -> None:
        """
        Plots individual topic-score 2D projections and MDS of average dissimilarity matrix.
        Connects scores with lines and color-codes dots based on score.
    
        Parameters
        ----------
        all_topics_points_2d : List[np.ndarray]
            A list of MDS projections, each of shape (num_ratings, 2).
        avg_points : np.ndarray
            (num_ratings, 2) array for the average dissimilarity MDS projection.
        num_ratings : int
            Number of ratings per topic/phrasing block.
        selected_phrasings : Optional[List[int]], default=None
            List of phrasing indices (0-based) to include in the plot.
            If None, all phrasings are plotted.
        categorical : bool, default=False
            If True, uses categorical labels for scores.
        """
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap
        from matplotlib.colors import Normalize
    
        # Validate categorical labels
        if categorical and not hasattr(self, 'scores'):
            raise AttributeError("`self.scores` must be defined for categorical plotting.")
        
        # Define a sequential color map
        cmap = get_cmap('viridis')
        norm = Normalize(vmin=0, vmax=len(self.scores)-1)
        colors = {score: cmap(norm(i)) for i, score in enumerate(self.scores)}
        
        plt.figure(figsize=(10, 6))
        
        # Default to all phrasings if none selected
        if not selected_phrasings:
            selected_phrasings = list(range(len(self.scores)))  # Adjust based on num_phrasings
        
        # Plot individual topic-score projections
        for idx, points in enumerate(all_topics_points_2d):
            phrasing_idx = idx % len(selected_phrasings)
            if phrasing_idx not in selected_phrasings:
                continue
            
            # Connect points with dark grey lines
            # plt.plot(points[:, 0], points[:, 1], color='darkgrey', linewidth=1, alpha=0.5)
            
            # Plot each score with corresponding color
            for i in range(num_ratings):
                score = self.scores[i]
                plt.scatter(points[i, 0], points[i, 1],
                            color=colors[score],
                            s=50,  # Smaller size for individual points
                            alpha=0.3,
                            edgecolors='w', linewidth=0.5)

        
        # Connect average points with lines in order of scores
        plt.plot(avg_points[:, 0], avg_points[:, 1], color='grey', linewidth=2, linestyle='-', alpha=0.8)
        
        # Plot average dissimilarity points
        for i in range(num_ratings):
            score = self.scores[i]
            plt.scatter(avg_points[i, 0], avg_points[i, 1],
                        color=colors[score],
                        s=150,  # Larger size for average points
                        alpha=1.0,
                        edgecolors='k', linewidth=1.5)
            # Offset the labels to avoid overlapping the dots
            offset_x = 0.02 * (plt.xlim()[1] - plt.xlim()[0])
            offset_y = 0.02 * (plt.ylim()[1] - plt.ylim()[0])
            plt.text(avg_points[i, 0] + offset_x, avg_points[i, 1] + offset_y, 
                     score if categorical else str(score),
                     fontsize=12, fontweight='bold',
                     ha='left', va='bottom', color='black',
                     bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.2'))
        

        
        # Create custom legend for scores
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=score,
                                  markerfacecolor=colors[score], markersize=10)
                           for score in self.scores]
        plt.legend(handles=legend_elements, title='Scores', loc='best')
        
        plt.title("MDS Projection of Ratings Across Topics/Phrasings", fontsize=16)
        plt.xlabel("Dimension 1", fontsize=14)
        plt.ylabel("Dimension 2", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.axis('off')
        plt.show()
