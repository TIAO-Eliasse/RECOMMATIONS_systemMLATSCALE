"""
Module for ALS Recommendation Model Integration
This file integrates the trained ALS model with optimized functions

Author: TIAO Eliasse, AIMS Student
🔧 Fixed: movie_id vs index mapping issue
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import pickle
import os
from numba import njit


# ============================
# OPTIMIZED NUMBA FUNCTIONS
# ============================

@njit
def fit_new_user_optimized(movie_indices, ratings, item_factors, item_biases,
                          K, lambda_reg, tau_reg, num_iterations=50, eps=1e-8):
    """
    Optimized version using single loops and Numba JIT compilation.
    Fits a new user profile based on their ratings.

    Parameters
    ----------
    movie_indices : np.array
        Array of movie indices that the user rated
    ratings : np.array
        Array of ratings given by the user
    item_factors : np.array
        Item factor matrix (N x K)
    item_biases : np.array
        Item bias vector
    K : int
        Number of latent factors
    lambda_reg, tau_reg : float
        Regularization parameters
    num_iterations : int
        Number of iterations for optimization
    eps : float
        Small constant for numerical stability

    Returns
    -------
    tuple
        (user_factor, user_bias)
    """
    user_bias = 0.0
    user_factor = np.random.normal(0, 0.1, K).astype(np.float32)

    n = len(movie_indices)
    I = np.eye(K, dtype=np.float32)

    # Extract relevant item factors and biases
    V_m = np.zeros((n, K), dtype=np.float32)
    b_i_m = np.zeros(n, dtype=np.float32)

    for i in range(n):
        idx = movie_indices[i]
        b_i_m[i] = item_biases[idx]
        for k in range(K):
            V_m[i, k] = item_factors[idx, k]

    for iteration in range(num_iterations):
        # Update user bias
        preds = np.zeros(n, dtype=np.float32)
        for i in range(n):
            # Compute dot product
            dot_prod = 0.0
            for k in range(K):
                dot_prod += V_m[i, k] * user_factor[k]
            preds[i] = dot_prod

        residuals_sum = 0.0
        for i in range(n):
            residuals_sum += ratings[i] - preds[i] - b_i_m[i]

        num = lambda_reg * residuals_sum
        den = lambda_reg * n + tau_reg
        user_bias = num / (den + eps)

        # Update user factor
        r_adj = np.zeros(n, dtype=np.float32)
        for i in range(n):
            r_adj[i] = ratings[i] - user_bias - b_i_m[i]

        # A = lambda_reg * (V_m.T @ V_m) + tau_reg * I + eps * I
        A = np.zeros((K, K), dtype=np.float32)
        for i in range(K):
            for j in range(K):
                # Compute V_m.T @ V_m
                for idx in range(n):
                    A[i, j] += lambda_reg * V_m[idx, i] * V_m[idx, j]
                # Add regularization
                if i == j:
                    A[i, j] += tau_reg + eps

        # b = lambda_reg * (V_m.T @ r_adj)
        b = np.zeros(K, dtype=np.float32)
        for k in range(K):
            for idx in range(n):
                b[k] += lambda_reg * V_m[idx, k] * r_adj[idx]

        user_factor = np.linalg.solve(A, b)

    return user_factor, user_bias


@njit
def predict_all_for_user_optimized(user_factor, item_factors, user_bias, item_biases, weight):
    """
    Returns a vector of predictions for all items for a given user.
    Uses single loops for efficiency with weighted item bias.

    Parameters
    ----------
    user_factor : np.array
        User factor vector (K,)
    item_factors : np.array
        Item factor matrix (N, K)
    user_bias : float
        User bias
    item_biases : np.array
        Item bias vector (N,)
    weight : float
        Weight for item biases (typically 0.05)

    Returns
    -------
    np.array
        Predictions for all items
    """
    N = item_factors.shape[0]
    K = item_factors.shape[1]
    predictions = np.zeros(N, dtype=np.float32)

    for n in range(N):
        # Compute dot product
        dot_prod = 0.0
        for k in range(K):
            dot_prod += user_factor[k] * item_factors[n, k]
        # Apply weighted item bias (weight = 0.05)
        predictions[n] = dot_prod + weight * item_biases[n]
    
    return predictions


# ============================
# MAIN CLASS
# ============================

class ALSRecommenderModel:
    """
    Class to manage ALS recommendation model
    Integration with trained and saved model
    
    🔧 FIXED: Proper handling of movie_id vs matrix index mapping
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize ALS model
        
        Args:
            model_path: Path to saved model (pickle)
        """
        self.model_data = None
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.movie_to_index = {}  # movie_id (int) -> matrix_index (int)
        self.index_to_movie = {}  # matrix_index (int) -> movie_id (int)
        self.user_to_index = {}
        self.index_to_user = {}
        self.hyperparameters = {}
        self.is_trained = False
        self.weight = 0.05  # Weight for item biases as specified
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load a pre-trained ALS model
        
        Args:
            model_path: Path to model file
        """
        try:
            with open(model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            
            # Extract model components
            self.hyperparameters = self.model_data.get('hyperparameters', {})
            model_params = self.model_data.get('model_parameters', {})
            mappings = self.model_data.get('mappings', {})
            
            # Factors and biases
            self.user_factors = model_params.get('user_factors')
            self.item_factors = model_params.get('item_factors')
            self.user_biases = model_params.get('user_biases')
            self.item_biases = model_params.get('item_biases')
            
            # 🔧 FIX: Ensure mappings use integer keys
            raw_movie_to_index = mappings.get('movie_to_index', {})
            raw_index_to_movie = mappings.get('index_to_movie', {})
            
            # Convert all keys to integers
            self.movie_to_index = {int(k): int(v) for k, v in raw_movie_to_index.items()}
            self.index_to_movie = {int(k): int(v) for k, v in raw_index_to_movie.items()}
            
            self.user_to_index = mappings.get('user_to_index', {})
            self.index_to_user = mappings.get('index_to_user', {})
            
            self.is_trained = True
            
            print(f"✅ Model loaded successfully from {model_path}")
            print(f"  - Number of users: {self.hyperparameters.get('M', 'N/A')}")
            print(f"  - Number of movies: {self.hyperparameters.get('N', 'N/A')}")
            print(f"  - Latent factors (K): {self.hyperparameters.get('K', 'N/A')}")
            print(f"  - Lambda: {self.hyperparameters.get('lambda_reg', 'N/A')}")
            print(f"  - Gamma: {self.hyperparameters.get('gamma_reg', 'N/A')}")
            print(f"  - Tau: {self.hyperparameters.get('tau_reg', 'N/A')}")
            print(f"  - Weight for item biases: {self.weight}")
            print(f"  - Movie IDs range: {min(self.movie_to_index.keys())} to {max(self.movie_to_index.keys())}")
        
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.is_trained = False
            raise
    
    def fit_new_user(self, user_ratings: Dict[int, float], 
                    num_iterations: int = 50) -> Tuple[np.ndarray, float]:
        """
        Compute profile for a new user based on their ratings
        
        🔧 FIXED: Now properly filters ratings to only use movies in model
        
        Args:
            user_ratings: Dictionary {movie_id: rating} of movies rated by user
            num_iterations: Number of iterations for optimization
            
        Returns:
            Tuple (user_factor, user_bias)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained.")
        
        # 🔧 FIX: Filter ratings to only include movies that exist in model
        valid_ratings = {}
        invalid_count = 0
        
        for movie_id, rating in user_ratings.items():
            # Ensure movie_id is integer
            movie_id = int(movie_id)
            
            if movie_id in self.movie_to_index:
                valid_ratings[movie_id] = rating
            else:
                invalid_count += 1
        
        # Debug info
        if invalid_count > 0:
            print(f"⚠️  {invalid_count} rated movies not in model (filtered out)")
        
        if not valid_ratings:
            print("⚠️  No valid ratings found! Returning default user profile.")
            # Return default user profile instead of raising error
            default_factor = np.random.normal(0, 0.1, self.hyperparameters['K']).astype(np.float32)
            return default_factor, 0.0
        
        print(f"✅ Using {len(valid_ratings)} valid ratings for user profile")
        
        # Convert movie_ids to indices
        movie_indices = []
        ratings = []
        
        for movie_id, rating in valid_ratings.items():
            idx = self.movie_to_index[movie_id]
            movie_indices.append(idx)
            ratings.append(rating)
        
        # Convert to numpy arrays
        movie_indices_arr = np.array(movie_indices, dtype=np.int32)
        ratings_arr = np.array(ratings, dtype=np.float32)
        
        # Use optimized function
        user_factor, user_bias = fit_new_user_optimized(
            movie_indices_arr,
            ratings_arr,
            self.item_factors,
            self.item_biases,
            self.hyperparameters['K'],
            self.hyperparameters['lambda_reg'],
            self.hyperparameters['tau_reg'],
            num_iterations=num_iterations
        )
        
        return user_factor, user_bias
    
    def predict_for_user(self, user_ratings: Dict[int, float], 
                        candidate_movie_ids: Optional[List[int]] = None) -> Dict[int, float]:
        """
        Predict ratings for a new user
        
        Args:
            user_ratings: Dictionary {movie_id: rating} of movies rated by user
            candidate_movie_ids: List of movie IDs to predict for (if None, all movies)
            
        Returns:
            Dictionary {movie_id: predicted_rating}
        """
        if not self.is_trained:
            raise ValueError("Model is not trained.")
        
        # Compute user profile
        user_factor, user_bias = self.fit_new_user(user_ratings)
        
        # Predict for all movies
        all_predictions = predict_all_for_user_optimized(
            user_factor,
            self.item_factors,
            user_bias,
            self.item_biases,
            self.weight
        )
        
        # If candidate_movie_ids is specified, filter predictions
        if candidate_movie_ids is not None:
            predictions = {}
            for movie_id in candidate_movie_ids:
                movie_id = int(movie_id)  # Ensure integer
                if movie_id in self.movie_to_index:
                    idx = self.movie_to_index[movie_id]
                    predictions[movie_id] = float(all_predictions[idx])
        else:
            # Return all predictions
            predictions = {}
            for idx, pred in enumerate(all_predictions):
                movie_id = self.index_to_movie.get(idx)
                if movie_id is not None:
                    predictions[movie_id] = float(pred)
        
        return predictions
    
    def get_recommendations(self, user_ratings: Dict[int, float],
                          all_movie_ids: Optional[List[int]] = None,
                          n_recommendations: int = 10,
                          exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        Get recommendations for a user
        
        Args:
            user_ratings: User ratings {movie_id: rating}
            all_movie_ids: List of all available movie IDs (if None, all movies in model)
            n_recommendations: Number of recommendations to return
            exclude_rated: If True, exclude already rated movies
            
        Returns:
            List of tuples (movie_id, predicted_rating) sorted by descending rating
        """
        # Determine candidate movies
        if all_movie_ids is None:
            all_movie_ids = list(self.movie_to_index.keys())
        
        # 🔧 FIX: Ensure all_movie_ids are integers
        all_movie_ids = [int(mid) for mid in all_movie_ids]
        
        # Filter candidate movies
        if exclude_rated:
            user_rating_ids = set(int(mid) for mid in user_ratings.keys())
            candidate_ids = [mid for mid in all_movie_ids if mid not in user_rating_ids]
        else:
            candidate_ids = all_movie_ids
        
        # 🔧 FIX: Only predict for movies that exist in model
        candidate_ids = [mid for mid in candidate_ids if mid in self.movie_to_index]
        
        if not candidate_ids:
            print("⚠️  No valid candidate movies found!")
            return []
        
        # Get predictions
        predictions = self.predict_for_user(user_ratings, candidate_ids)
        
        # Sort by predicted rating descending
        sorted_predictions = sorted(predictions.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)
        
        return sorted_predictions[:n_recommendations]
    
    def get_similar_movies(self, movie_id: int, 
                          n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Find similar movies based on latent factors
        
        Args:
            movie_id: Reference movie ID
            n_similar: Number of similar movies to return
            
        Returns:
            List of tuples (movie_id, similarity_score)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained.")
        
        movie_id = int(movie_id)  # Ensure integer
        
        if movie_id not in self.movie_to_index:
            raise ValueError(f"Movie {movie_id} is not in model.")
        
        # Get movie index
        movie_idx = self.movie_to_index[movie_id]
        
        # Calculate similarities (cosine similarity)
        movie_vector = self.item_factors[movie_idx]
        similarities = []
        
        for idx in range(len(self.item_factors)):
            if idx != movie_idx:
                other_vector = self.item_factors[idx]
                # Cosine similarity
                similarity = np.dot(movie_vector, other_vector) / (
                    np.linalg.norm(movie_vector) * np.linalg.norm(other_vector) + 1e-10
                )
                other_movie_id = self.index_to_movie[idx]
                similarities.append((other_movie_id, float(similarity)))
        
        # Sort by descending similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n_similar]


# ============================
# INTEGRATION FUNCTIONS
# ============================

def hybrid_recommendation(user_ratings: Dict[int, float],
                         als_model: ALSRecommenderModel,
                         df_movies: pd.DataFrame,
                         n_recommendations: int = 10,
                         als_weight: float = 0.7,
                         content_weight: float = 0.3) -> List[int]:
    """
    Hybrid recommendation system combining ALS and content filtering
    
    ⚠️ IMPORTANT: Parameter order changed for consistency
    
    Args:
        user_ratings: User ratings {movie_id: rating}
        als_model: Trained ALS model
        df_movies: Movies DataFrame
        n_recommendations: Number of recommendations to return
        als_weight: Weight for ALS score (not used for now)
        content_weight: Weight for content score (not used for now)
        
    Returns:
        List of recommended movie IDs, sorted by descending score
        
    Example:
        >>> ratings = {1: 5.0, 2: 4.0, 3: 3.5}
        >>> model = ALSRecommenderModel('model.pkl')
        >>> recs = hybrid_recommendation(ratings, model, df_movies, 10)
        >>> print(recs)  # [123, 456, 789, ...]
    """
    if not als_model.is_trained:
        # Fallback to simple collaborative recommendations
        from utils import get_collaborative_recommendations
        return get_collaborative_recommendations(user_ratings, df_movies, n_recommendations)
    
    try:
        # Get ALS recommendations based on user ratings
        all_movie_ids = df_movies['movieId'].tolist()
        
        # ✅ The ALS model automatically computes user profile
        # based on user_ratings and predicts for all movies
        als_recs = als_model.get_recommendations(
            user_ratings=user_ratings,
            all_movie_ids=all_movie_ids,
            n_recommendations=n_recommendations,
            exclude_rated=True
        )
        
        # Return recommended movie IDs (sorted by descending score)
        return [movie_id for movie_id, _ in als_recs]
    
    except Exception as e:
        print(f"⚠️  Error in hybrid_recommendation: {e}")
        import traceback
        traceback.print_exc()
        # Fallback
        from utils import get_collaborative_recommendations
        return get_collaborative_recommendations(user_ratings, df_movies, n_recommendations)


# ============================
# USAGE EXAMPLE
# ============================

if __name__ == "__main__":
    print("=" * 80)
    print("ALS MODEL INTEGRATION MODULE")
    print("=" * 80)
    
    print("\n📋 This module contains:")
    print("   ✅ Loading trained ALS model")
    print("   ✅ Computing profile for new users (fit_new_user_optimized)")
    print("   ✅ Predictions for all movies (predict_all_for_user_optimized)")
    print("   ✅ Personalized recommendations")
    print("   ✅ Similar movies based on latent factors")
    print("   ✅ Hybrid recommendation")
    
    print("\n⚙️  Configuration:")
    print("   - Weight for item biases: 0.05")
    print("   - Optimized functions with Numba JIT")
    
    print("\n🔧 To use the model:")
    print("   1. Load: model = ALSRecommenderModel('path/to/model.pkl')")
    print("   2. Predict: predictions = model.predict_for_user(user_ratings)")
    print("   3. Recommend: recs = model.get_recommendations(user_ratings, n=10)")
    
    print("\n🔧 FIXED: Proper movie_id to matrix index mapping!")
    print("   - All movie IDs are now properly converted to integers")
    print("   - Invalid ratings are automatically filtered out")
    print("   - Graceful fallback if no valid ratings found")
    
    print("\n" + "=" * 80)