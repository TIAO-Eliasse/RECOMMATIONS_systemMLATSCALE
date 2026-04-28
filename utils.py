"""
Fichier de fonctions utilitaires pour le système de recommandation de films
"""

import pandas as pd
import requests
from typing import List, Dict, Optional, Tuple
import time
from functools import lru_cache


# ============================
# CONFIGURATION API
# ============================

# TMDB API (The Movie Database - Gratuit avec inscription)
# Pour obtenir une clé : https://www.themoviedb.org/settings/api
TMDB_API_KEY = "YOUR_TMDB_API_KEY"  # À remplacer par votre clé
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# OMDb API (Alternative - Gratuit avec limite)
# Pour obtenir une clé : http://www.omdbapi.com/apikey.aspx
OMDB_API_KEY = "YOUR_OMDB_API_KEY"  # À remplacer par votre clé
OMDB_BASE_URL = "http://www.omdbapi.com/"


# ============================
# CHARGEMENT DES DONNÉES
# ============================

@lru_cache(maxsize=1)
def load_movies_data(file_path: str = "movies.csv") -> pd.DataFrame:
    """
    Charge le fichier CSV des films
    
    Args:
        file_path: Chemin vers le fichier movies.csv
        
    Returns:
        DataFrame avec les informations des films
    """
    try:
        df = pd.read_csv(file_path)
        # Extraire l'année du titre si elle est entre parenthèses
        df['year'] = df['title'].str.extract(r'\((\d{4})\)')
        df['clean_title'] = df['title'].str.replace(r'\s*\(\d{4}\)\s*', '', regex=True)
        return df
    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} n'a pas été trouvé.")
        return pd.DataFrame()


def get_movie_by_id(movie_id: int, df: pd.DataFrame) -> Optional[Dict]:
    """
    Récupère les informations d'un film par son ID
    
    Args:
        movie_id: ID du film
        df: DataFrame contenant les films
        
    Returns:
        Dictionnaire avec les informations du film ou None
    """
    movie = df[df['movieId'] == movie_id]
    if not movie.empty:
        return movie.iloc[0].to_dict()
    return None


def search_movies(query: str, df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """
    Recherche des films par titre
    
    Args:
        query: Terme de recherche
        df: DataFrame contenant les films
        limit: Nombre maximum de résultats
        
    Returns:
        DataFrame avec les films correspondants
    """
    if not query:
        return pd.DataFrame()
    
    query_lower = query.lower()
    mask = df['title'].str.lower().str.contains(query_lower, na=False)
    return df[mask].head(limit)


# ============================
# RÉCUPÉRATION D'IMAGES DE FILMS
# ============================

@lru_cache(maxsize=200)
def get_movie_poster_tmdb(title: str, year: Optional[str] = None) -> Optional[str]:
    """
    Récupère l'URL du poster d'un film via TMDB API
    
    Args:
        title: Titre du film
        year: Année de sortie (optionnel)
        
    Returns:
        URL du poster ou None
    """
    if TMDB_API_KEY == "YOUR_TMDB_API_KEY":
        return None
    
    try:
        # Recherche du film
        search_url = f"{TMDB_BASE_URL}/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": title,
            "language": "fr-FR"
        }
        
        if year:
            params["year"] = year
        
        response = requests.get(search_url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("results"):
                poster_path = data["results"][0].get("poster_path")
                if poster_path:
                    return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
        
        return None
    
    except Exception as e:
        print(f"Erreur TMDB pour {title}: {e}")
        return None


@lru_cache(maxsize=200)
def get_movie_poster_omdb(title: str, year: Optional[str] = None) -> Optional[str]:
    """
    Récupère l'URL du poster d'un film via OMDb API
    
    Args:
        title: Titre du film
        year: Année de sortie (optionnel)
        
    Returns:
        URL du poster ou None
    """
    if OMDB_API_KEY == "YOUR_OMDB_API_KEY":
        return None
    
    try:
        params = {
            "apikey": OMDB_API_KEY,
            "t": title,
        }
        
        if year:
            params["y"] = year
        
        response = requests.get(OMDB_BASE_URL, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            poster = data.get("Poster")
            if poster and poster != "N/A":
                return poster
        
        return None
    
    except Exception as e:
        print(f"Erreur OMDb pour {title}: {e}")
        return None


def get_movie_poster(title: str, year: Optional[str] = None, 
                     use_tmdb: bool = True, use_omdb: bool = True) -> Optional[str]:
    """
    Récupère l'URL du poster d'un film en essayant plusieurs API
    
    Args:
        title: Titre du film
        year: Année de sortie (optionnel)
        use_tmdb: Utiliser TMDB API
        use_omdb: Utiliser OMDb API
        
    Returns:
        URL du poster ou None
    """
    # Nettoyer le titre
    clean_title = title.split('(')[0].strip() if '(' in title else title
    
    # Essayer TMDB d'abord
    if use_tmdb:
        poster = get_movie_poster_tmdb(clean_title, year)
        if poster:
            return poster
        time.sleep(0.1)  # Rate limiting
    
    # Essayer OMDb si TMDB échoue
    if use_omdb:
        poster = get_movie_poster_omdb(clean_title, year)
        if poster:
            return poster
    
    # Retourner une image par défaut
    return "https://via.placeholder.com/500x750?text=No+Image"


def get_movies_with_posters(movie_ids: List[int], df: pd.DataFrame) -> List[Dict]:
    """
    Récupère les informations et posters pour une liste de films
    
    Args:
        movie_ids: Liste des IDs de films
        df: DataFrame contenant les films
        
    Returns:
        Liste de dictionnaires avec informations et posters
    """
    movies_with_posters = []
    
    for movie_id in movie_ids:
        movie = get_movie_by_id(movie_id, df)
        if movie:
            year = movie.get('year')
            poster_url = get_movie_poster(movie['title'], year)
            
            movies_with_posters.append({
                'movieId': movie['movieId'],
                'title': movie['title'],
                'genres': movie['genres'],
                'year': year,
                'poster_url': poster_url
            })
            
            # Rate limiting pour éviter de surcharger les API
            time.sleep(0.2)
    
    return movies_with_posters


# ============================
# GESTION DES NOTES UTILISATEUR
# ============================

class UserRatingsManager:
    """Gestionnaire des notes utilisateur"""
    
    def __init__(self):
        self.ratings = {}  # {movie_id: rating}
    
    def add_rating(self, movie_id: int, rating: float):
        """Ajoute ou met à jour une note"""
        self.ratings[movie_id] = rating
    
    def remove_rating(self, movie_id: int):
        """Supprime une note"""
        if movie_id in self.ratings:
            del self.ratings[movie_id]
    
    def get_rating(self, movie_id: int) -> Optional[float]:
        """Récupère la note d'un film"""
        return self.ratings.get(movie_id)
    
    def get_all_ratings(self) -> Dict[int, float]:
        """Récupère toutes les notes"""
        return self.ratings.copy()
    
    def get_rated_movies_count(self) -> int:
        """Retourne le nombre de films notés"""
        return len(self.ratings)
    
    def clear_ratings(self):
        """Efface toutes les notes"""
        self.ratings.clear()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convertit les notes en DataFrame"""
        if not self.ratings:
            return pd.DataFrame(columns=['movieId', 'rating'])
        
        return pd.DataFrame([
            {'movieId': movie_id, 'rating': rating}
            for movie_id, rating in self.ratings.items()
        ])


# ============================
# SYSTÈME DE RECOMMANDATION
# ============================

def get_collaborative_recommendations(
    user_ratings: Dict[int, float],
    df_movies: pd.DataFrame,
    n_recommendations: int = 10
) -> List[int]:
    """
    Génère des recommandations basées sur le filtrage collaboratif
    PLACEHOLDER - À remplacer par votre modèle ALS
    
    Args:
        user_ratings: Dictionnaire {movie_id: rating}
        df_movies: DataFrame des films
        n_recommendations: Nombre de recommandations
        
    Returns:
        Liste des IDs de films recommandés
    """
    # Pour l'instant, recommandation simple basée sur les genres
    # VOUS DEVREZ REMPLACER CECI PAR VOTRE MODÈLE ALS
    
    if not user_ratings:
        # Si pas de notes, retourner les films populaires
        return df_movies.head(n_recommendations)['movieId'].tolist()
    
    # Extraire les genres préférés
    rated_movie_ids = list(user_ratings.keys())
    rated_movies = df_movies[df_movies['movieId'].isin(rated_movie_ids)]
    
    # Compter les genres
    all_genres = []
    for genres_str in rated_movies['genres']:
        if pd.notna(genres_str):
            all_genres.extend(genres_str.split('|'))
    
    # Trouver les genres les plus fréquents
    from collections import Counter
    genre_counts = Counter(all_genres)
    top_genres = [genre for genre, _ in genre_counts.most_common(3)]
    
    # Trouver des films non notés avec ces genres
    unrated_movies = df_movies[~df_movies['movieId'].isin(rated_movie_ids)]
    
    recommendations = []
    for _, movie in unrated_movies.iterrows():
        if pd.notna(movie['genres']):
            movie_genres = movie['genres'].split('|')
            if any(genre in top_genres for genre in movie_genres):
                recommendations.append(movie['movieId'])
                if len(recommendations) >= n_recommendations:
                    break
    
    # Compléter avec des films aléatoires si nécessaire
    if len(recommendations) < n_recommendations:
        remaining = unrated_movies[~unrated_movies['movieId'].isin(recommendations)]
        additional = remaining.head(n_recommendations - len(recommendations))['movieId'].tolist()
        recommendations.extend(additional)
    
    return recommendations[:n_recommendations]


def get_content_based_recommendations(
    movie_id: int,
    df_movies: pd.DataFrame,
    n_recommendations: int = 10
) -> List[int]:
    """
    Génère des recommandations basées sur le contenu (genres similaires)
    
    Args:
        movie_id: ID du film de référence
        df_movies: DataFrame des films
        n_recommendations: Nombre de recommandations
        
    Returns:
        Liste des IDs de films recommandés
    """
    movie = get_movie_by_id(movie_id, df_movies)
    
    if not movie or pd.isna(movie.get('genres')):
        return []
    
    movie_genres = set(movie['genres'].split('|'))
    
    # Calculer la similarité avec les autres films
    similarities = []
    for _, other_movie in df_movies.iterrows():
        if other_movie['movieId'] == movie_id:
            continue
        
        if pd.notna(other_movie['genres']):
            other_genres = set(other_movie['genres'].split('|'))
            # Similarité de Jaccard
            similarity = len(movie_genres & other_genres) / len(movie_genres | other_genres)
            similarities.append((other_movie['movieId'], similarity))
    
    # Trier par similarité
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return [movie_id for movie_id, _ in similarities[:n_recommendations]]


# ============================
# UTILITAIRES D'AFFICHAGE
# ============================

def format_genres(genres: str) -> str:
    """
    Formate la chaîne de genres pour l'affichage
    
    Args:
        genres: Chaîne de genres séparés par |
        
    Returns:
        Genres formatés
    """
    if pd.isna(genres):
        return "Non spécifié"
    return genres.replace('|', ' • ')


def get_movie_description(movie: Dict) -> str:
    """
    Génère une description pour un film
    
    Args:
        movie: Dictionnaire avec les infos du film
        
    Returns:
        Description formatée
    """
    title = movie.get('title', 'Titre inconnu')
    genres = format_genres(movie.get('genres', ''))
    year = movie.get('year', '')
    
    if year:
        return f"**{title}**\n\n🎭 {genres}\n📅 {year}"
    else:
        return f"**{title}**\n\n🎭 {genres}"
