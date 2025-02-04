import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Optional, Dict
from tqdm import tqdm

# For embeddings
import openai
import cohere
import google.generativeai as genai
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


class MDSProjector:
    """
    A class that orchestrates sentence generation, embedding retrieval, MDS projection, and result plotting.
    Designed for both numeric (1–10) and categorical rating scales.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key  # for OpenAI or others
        self.scores = list(range(1, 11))  # default numeric scores

    def generate_sentences(
        self, 
        concepts: List[str], 
        categorical: bool = False, 
        num_phrasings: int = 10
    ) -> List[dict]:
        """
        Generates multiple sentences for each concept with phrasing variations.
        Supports both categorical and numerical scoring systems.
        """
        if categorical:
            # self.scores = [ "F", "D−", "D", ... "A+" ] (whatever you want)
            # self.scores = [ "F", "D−", "D", "D+","C−", "C", "C+", "B−", "B", "B+","A−","A","A+"]
            self.scores = ["dreadful", "terrible","inadequate", "mediocre", "acceptable", 
            "decent", "good", "great", "excellent", "superb", "magnificent", "perfect"]
            phrasings = [
                lambda concept, score: f"We are {concept}, my evaluation is {score}",
                lambda concept, score: f"Rated as {score} after {concept}",
                lambda concept, score: f"After {concept}, receiving a(n) {score} rating",
                lambda concept, score: f"On the topic of {concept}, my score is {score}",
                lambda concept, score: f"{concept.capitalize()} gets rated as {score}",
                lambda concept, score: f"On {concept}, judgement is that its {score}",
                lambda concept, score: f"{concept.capitalize()} is evaluated as {score}",
                lambda concept, score: f"The rating for {concept} is {score}",
                lambda concept, score: f"{concept.capitalize()} scores as {score}",
                lambda concept, score: f"In terms of {concept}, my score is {score}",
            ][:num_phrasings]
        else:
            # Numeric scale
            self.scores = list(range(1, 11))  # 1..10 or 1..10 step=1
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

    def get_embeddings(self, sentences: List[Union[str, dict]], model_name: str, OPENAI_APIKEY: str) -> np.ndarray:
        """
        Obtain embeddings for a list of sentences from various models (OpenAI, HF, etc.).
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
        elif model_name.startswith("cohere"):
            return self._get_cohere_embeddings(texts, model_name.split('/')[-1])
        elif model_name.startswith("gemini"):
            return self._get_gemini_embeddings(texts, model_name.split('/', 1)[-1])
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



    def _get_cohere_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """
        Retrieve embeddings from Cohere in a sequential fashion (one text at a time).
        
        Parameters:
        - texts: List of input strings to embed.
        - model_name: Name of the Cohere embedding model.
        
        Returns:
        - np.ndarray: A NumPy array of shape (num_texts, embedding_dim).
        """
        import time
        # Initialize Cohere client
        co = cohere.ClientV2(api_key=os.getenv("APIKEY_COHERE"))

        embeddings = []
        
        for txt in tqdm(texts, desc=f"Fetching Cohere ({model_name}) embeddings"):
            try:
                response = co.embed(
                    texts=[txt],  # Process one text at a time
                    model=model_name,
                    input_type="classification",
                    embedding_types=["float"],
                )

                emb = response.embeddings  # Extract embeddings
                
                # Ensure proper unwrapping if wrapped in `float_`
                if hasattr(emb, "float_"):
                    emb = emb.float_

                embeddings.append(emb[0])  # Extract the first (and only) embedding
                time.sleep(0.7)
            except Exception as e:
                print(f"Error for '{txt[:50]}...': {e}")
                # Use zero vector matching the expected dimension (assume 1024)
                dim = 1024  # Adjust if needed
                embeddings.append([0.0] * dim)

        return np.array(embeddings, dtype=np.float32)

    def _get_gemini_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """
        Retrieve embeddings from Google's Gemini API in a sequential fashion.
        
        Parameters:
        - texts: List of input strings to embed.
        - model_name: Name of the Gemini embedding model (e.g., "models/text-embedding-004").
        
        Returns:
        - np.ndarray: A NumPy array of shape (num_texts, embedding_dim).
        """
        # Configure the API key
        genai.configure(api_key=os.getenv("APIKEY_GEMINI"))

        embeddings = []
        
        for txt in tqdm(texts, desc=f"Fetching Gemini ({model_name}) embeddings"):
            try:
                response = genai.embed_content(
                    model=model_name,
                    content=txt
                )
                
                emb = response['embedding']  # Extract the embedding

                embeddings.append(emb)  # Append the embedding

            except Exception as e:
                print(f"Error for '{txt[:50]}...': {e}")
                # Use zero vector matching the expected dimension (assume 768, adjust as needed)
                dim = 768  
                embeddings.append([0.0] * dim)

        return np.array(embeddings, dtype=np.float32)


    def _get_voyage_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """
        Retrieve embeddings from Voyage (pseudo-code).
        """
        if not self.api_key:
            raise ValueError("Voyage API key is not provided.")
        # For demonstration, return random
        return np.random.rand(len(texts), 1024).astype(np.float32)

    def _get_claude_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """
        Retrieve embeddings from Claude (pseudo-code).
        """
        if not self.api_key:
            raise ValueError("Anthropic/Claude API key is not provided.")
        # For demonstration, return random
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

    def save_embeddings(self, embeddings: np.ndarray, filepath: str = "embeddings.pkl") -> None:
        """
        Old method that saves *only* the raw embeddings.
        """
        with open(filepath, "wb") as f:
            pickle.dump(embeddings, f)

    def load_embeddings(self, filepath: str = "embeddings.pkl") -> np.ndarray:
        """
        Old method that loads *only* the raw embeddings.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Embedding file not found: {filepath}")

        with open(filepath, "rb") as f:
            return pickle.load(f)

    ##
    ## NEW METHODS FOR STORING/LOADING FULL CONFIG + EMBEDDINGS
    ##

    def save_data(self, data: Dict, filepath: str) -> None:
        """
        Saves an entire dictionary (e.g. with embeddings + config + sentences).
        """
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load_data(self, filepath: str) -> Dict:
        """
        Loads an entire dictionary from file (with embeddings + config + sentences).
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def load_or_create_embeddings(
        self,
        concepts: List[str],
        model_name: str,
        rating_type: str = "continuous",
        num_phrasings: int = 3,
        base_dir: str = "ratings",
        openai_api_key: str = ""
    ) -> Dict:
        """
        1) Constructs a filename from the config.
        2) If that file exists, loads it.
        3) If not, generates sentences, obtains embeddings, and saves.

        Returns a dictionary:
            {
              'config': { ... },
              'embeddings': np.ndarray,
              'sentences': List[dict]
            }
        """
        # Distinguish 'categorical' vs 'continuous' in the filename
        filename_tag = "categorical" if rating_type == "categorical" else "continuous"
        # Build the name
        file_name = (
            f"{model_name.replace('/','-')}_{filename_tag}"
            f"_{len(concepts)}topics_{num_phrasings}phrasings.pkl"
        )
        filepath = os.path.join(base_dir, file_name)

        # If it exists, load it
        if os.path.exists(filepath):
            print(f"Loading existing embeddings from {filepath}")
            data = self.load_data(filepath)
            # Also restore self.scores from the data's "scores" if needed
            if "scores" in data["config"]:
                self.scores = data["config"]["scores"]
            return data

        print(f"File {filepath} not found; creating embeddings from scratch...")

        # Otherwise, generate everything
        categorical = (rating_type == "categorical")
        all_sentences = self.generate_sentences(concepts, categorical=categorical, num_phrasings=num_phrasings)
        embeddings = self.get_embeddings(all_sentences, model_name, openai_api_key)

        # Save relevant config and data
        data = {
            "config": {
                "concepts": concepts,
                "model_name": model_name,
                "rating_type": rating_type,
                "num_phrasings": num_phrasings,
                "scores": self.scores,
            },
            "sentences": all_sentences,
            "embeddings": embeddings,
        }

        # Save to disk
        os.makedirs(base_dir, exist_ok=True)
        self.save_data(data, filepath)
        print(f"Saved new embeddings + config => {filepath}")
        return data

    ##
    ## MDS + PLOTTING
    ##

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
        """
        if scale:
            scaler = StandardScaler()
            embeddings = scaler.fit_transform(embeddings)

        # Reduce dimensionality before computing similarities
        pca = PCA(n_components=2)  # Keep only top 50 dimensions (adjustable)
        embeddings_pca = pca.fit_transform(embeddings)

        # Compute similarity & dissimilarity after PCA
        sim_matrix = cosine_similarity(embeddings_pca)
        diss_matrix = 1.0 - sim_matrix

        mds = MDS(
            n_components=2,
            metric=True,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            dissimilarity='precomputed',
            eps=1e-4,
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
        Computes MDS for each topic-phrasing block, plus an MDS of the average dissimilarity.
        """
        from tqdm import tqdm
        from scipy.spatial import procrustes

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D: (n_samples, embedding_dim).")

        expected_size = num_topics * num_ratings * num_phrasings
        if embeddings.shape[0] != expected_size:
            raise ValueError(
                f"Expected {expected_size} embeddings, got {embeddings.shape[0]}."
            )

        # Reshape
        embedding_dim = embeddings.shape[1]
        reshaped = embeddings.reshape(num_topics, num_ratings, num_phrasings, embedding_dim)
        # Reorder
        reordered = reshaped.transpose(2, 0, 1, 3).reshape(num_phrasings * num_topics, num_ratings, embedding_dim)

        mds = MDS(
            n_components=2,
            metric=True,
            n_init=10,
            max_iter=1500,
            random_state=42,
            dissimilarity='precomputed',
            eps=0.3e-3,

        )

        all_topics_points_2d = []
        all_dissimilarity_matrices = []
        reference_2d = None

        for block in tqdm(reordered, desc="Computing MDS for each block"):
            sim_matrix = cosine_similarity(block)
            diss_matrix = 1.0 - sim_matrix
            np.fill_diagonal(diss_matrix, 0.0)
            all_dissimilarity_matrices.append(diss_matrix)

            points_2d = mds.fit_transform(diss_matrix)

            # Align with the reference block via Procrustes
            if reference_2d is None:
                reference_2d = points_2d.copy()
            else:
                _, aligned, _ = procrustes(reference_2d, points_2d)
                points_2d = aligned

            all_topics_points_2d.append(points_2d)

        avg_dissimilarity = np.mean(all_dissimilarity_matrices, axis=0)
        avg_points = mds.fit_transform(avg_dissimilarity)
        # Align average too
        _, avg_points_aligned, _ = procrustes(reference_2d, avg_points)
        avg_points = avg_points_aligned

        return all_topics_points_2d, avg_points, all_dissimilarity_matrices

    def plot_results(
        self,
        all_topics_points_2d: List[np.ndarray],
        avg_points: np.ndarray,
        num_ratings: int,
        num_phrasings: int,  # <---- add this
        selected_phrasings: Optional[List[int]] = None,
        categorical: bool = False
    ) -> None:
        """
        Plots each topic-score 2D projection and the average MDS projection.
        """
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap
        from matplotlib.colors import Normalize

        if categorical and not hasattr(self, 'scores'):
            raise AttributeError("`self.scores` must be defined for categorical plotting.")
        
        cmap = get_cmap('viridis')
        norm = Normalize(vmin=0, vmax=len(self.scores)-1)
        colors = {score: cmap(norm(i)) for i, score in enumerate(self.scores)}
        
        plt.figure(figsize=(10, 6))

        # But note: we have 'all_topics_points_2d' with length (num_phrasings * num_topics)
        # If none selected, select all
        if selected_phrasings is None:
            selected_phrasings = list(range(num_phrasings))

        for idx, points in enumerate(all_topics_points_2d):
            phrasing_idx = idx % num_phrasings
            if phrasing_idx not in selected_phrasings:
                continue


            # Faint connecting line between points
            plt.plot(points[:, 0], points[:, 1], color='grey', linewidth=1, alpha=0.07)

            # Plot points
            for i in range(num_ratings):
                score = self.scores[i]
                plt.scatter(points[i, 0], points[i, 1],
                            color=colors[score],
                            s=50,
                            alpha=0.35,
                            edgecolors='w', linewidth=0.5)



        # Connect average points
        plt.plot(avg_points[:, 0], avg_points[:, 1], color='grey', linewidth=2, linestyle='-', alpha=0.8)
        
        # Plot average dissimilarity points
        for i in range(num_ratings):
            score = self.scores[i]
            plt.scatter(avg_points[i, 0], avg_points[i, 1],
                        color=colors[score],
                        s=150,
                        alpha=1.0,
                        edgecolors='k', linewidth=1.5)
            # Label
            offset_x = 0.02 * (plt.xlim()[1] - plt.xlim()[0])
            offset_y = 0.02 * (plt.ylim()[1] - plt.ylim()[0])
            plt.text(avg_points[i, 0] + offset_x, avg_points[i, 1] + offset_y,
                     score if categorical else str(score),
                     fontsize=12, fontweight='bold',
                     ha='left', va='bottom', color='black',
                     bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.2'))

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=score,
                                  markerfacecolor=colors[score], markersize=10)
                           for score in self.scores]
        # plt.legend(handles=legend_elements, title='Scores', loc='best')

        plt.title("MDS Projection of Ratings Across Topics/Phrasings", fontsize=16)
        plt.xlabel("Dimension 1", fontsize=14)
        plt.ylabel("Dimension 2", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.axis('off')

        # Save the plot as PNG and SVG
        plt.savefig("outputs/mds_projection_categorical_v1.png", dpi=300, bbox_inches='tight', format='png')  # High-resolution PNG
        plt.savefig("outputs/mds_projection_categorical_v1.svg", bbox_inches='tight', format='svg')  # Vector-format SVG

        plt.show()
