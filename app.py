"""
Streamlit Application - Movie Recommendation System
TMDB-inspired interface with automatic poster download and external movie links

Author: TIAO Eliasse, AIMS Student
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Optional
import time
import requests
import os
from dotenv import load_dotenv
import re
import hashlib
import os
import zipfile
import pickle
import streamlit as st


# Load environment variables
load_dotenv()

# Import local modules
from utils import (
    load_movies_data,
    search_movies,
    UserRatingsManager,
    get_collaborative_recommendations,
    get_content_based_recommendations,
)
from model_integration import ALSRecommenderModel, hybrid_recommendation


# ============================
# TMDB API CONFIGURATION
# ============================

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_MOVIE_BASE_URL = "https://www.themoviedb.org/movie/"


# ============================
# PAGE CONFIGURATION
# ============================

st.set_page_config(
    page_title="Movie Recommender - TIAO Eliasse",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================
# TMDB-INSPIRED CSS (FIXED)
# ============================

def load_custom_css():
    """Loads TMDB-inspired CSS styling with fixes"""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
        
        * {
            font-family: 'Source Sans Pro', sans-serif;
        }
        
        .main {
            background-color: #0d253f;
            padding: 0 !important;
        }
        
        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        
        /* Hero Section */
        .hero-section {
            background: linear-gradient(to right, rgba(3, 37, 65, 0.95) 0%, rgba(3, 37, 65, 0.7) 100%), 
                        url('https://www.themoviedb.org/t/p/w1920_and_h600_multi_faces_filter(duotone,032541,01b4e4)/9n2tJBplPbgR2ca05hS5CKXwP2c.jpg');
            background-size: cover;
            background-position: center;
            padding: 60px 40px;
            color: white;
            margin-bottom: 0;
        }
        
        .hero-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .hero-left {
            flex: 1;
        }
        
        .hero-title {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 10px;
            color: white;
        }
        
        .hero-subtitle {
            font-size: 1.5rem;
            font-weight: 300;
            margin-bottom: 0;
            color: white;
        }
        
        .hero-stats {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        
        .hero-stat-item {
            background: rgba(1, 180, 228, 0.2);
            backdrop-filter: blur(10px);
            border: 2px solid rgba(1, 180, 228, 0.4);
            border-radius: 12px;
            padding: 15px 25px;
            color: white;
            text-align: center;
            min-width: 130px;
        }
        
        .hero-stat-number {
            font-size: 2rem;
            font-weight: 700;
            display: block;
            color: #01b4e4;
        }
        
        .hero-stat-label {
            font-size: 0.85rem;
            margin-top: 4px;
            display: block;
            opacity: 0.9;
        }
        
        /* Search Bar */
        .search-section {
            background-color: #032541;
            padding: 25px 40px;
            border-bottom: 1px solid rgba(1,180,228,0.2);
        }
        
        .stTextInput > div > div > input {
            background-color: white;
            border: none;
            border-radius: 30px;
            color: #000;
            padding: 18px 25px;
            font-size: 1.1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        /* Control Panel - REDUCED PADDING */
        .control-panel {
            background-color: #032541;
            padding: 20px 40px;
            margin-bottom: 0;
            border-bottom: 1px solid rgba(1,180,228,0.2);
        }
        
        /* Section Container - REDUCED PADDING */
        .section-container {
            padding: 30px 20px;
            background-color: #0d253f;
        }
        
        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(1,180,228,0.3);
        }
        
        .section-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: white;
            margin: 0;
        }
        
        /* Movie Card Container - NO BOTTOM MARGIN */
        .movie-card-wrapper {
            margin-bottom: 0px;
        }
        
        /* Movie Card */
        .movie-card {
            background-color: #fff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            cursor: pointer;
            margin-bottom: 20px;
        }
        
        .movie-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        
        .poster-wrapper {
            position: relative;
            width: 100%;
            padding-top: 150%;
            overflow: hidden;
            background: linear-gradient(135deg, #dbdbdb 0%, #c0c0c0 100%);
        }
        
        .poster-image {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .rating-badge {
            position: absolute;
            bottom: -20px;
            left: 10px;
            width: 40px;
            height: 40px;
            background-color: #081c22;
            border: 3px solid #21d07a;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.9rem;
            color: #21d07a;
            z-index: 10;
        }
        
        .movie-info {
            padding: 26px 10px 12px 10px;
            background-color: white;
        }
        
        .movie-title {
            font-size: 1rem;
            font-weight: 700;
            color: #000;
            margin-bottom: 4px;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
            line-height: 1.3;
            min-height: 2.6em;
        }
        
        .movie-date {
            font-size: 0.9rem;
            color: rgba(0,0,0,0.6);
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(to right, #01b4e4, #0d8fb8);
            color: white;
            border-radius: 30px;
            border: none;
            padding: 12px 28px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(1,180,228,0.5);
        }
        
        /* Tabs - DISTRIBUTED ACROSS FULL WIDTH */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background-color: #032541;
            padding: 0 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 100%;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 18px 40px;
            background-color: transparent;
            color: rgba(255,255,255,0.7);
            font-weight: 600;
            font-size: 1.05rem;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            color: rgba(255,255,255,0.9);
        }
        
        .stTabs [aria-selected="true"] {
            color: #01b4e4;
            border-bottom-color: #01b4e4;
        }
        
        /* Selectbox */
        .stSelectbox > div > div {
            background-color: rgba(255,255,255,0.15);
            border-radius: 8px;
            border: 1px solid rgba(1,180,228,0.4);
        }
        
        .stSelectbox label {
            color: white !important;
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        /* Rating Modal */
        .rating-modal {
            background-color: white;
            border-radius: 12px;
            padding: 25px;
            margin: 15px 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        }
        
        .rating-modal h3 {
            color: #032541;
            margin-bottom: 15px;
            font-size: 1.1rem;
        }
        
        /* Star Rating System */
        .star-rating-container {
            display: flex;
            gap: 8px;
            justify-content: center;
            margin: 20px 0;
        }
        
        .star-btn {
            background: none;
            border: none;
            font-size: 2.5rem;
            cursor: pointer;
            transition: transform 0.2s;
            padding: 0;
        }
        
        .star-btn:hover {
            transform: scale(1.2);
        }
        
        .star-filled {
            color: #FFD700;
            text-shadow: 0 0 5px rgba(255, 215, 0, 0.5);
        }
        
        .star-empty {
            color: #ddd;
        }
        
        .rating-value {
            text-align: center;
            color: #01b4e4;
            font-size: 1.2rem;
            font-weight: 600;
            margin-top: 10px;
        }
        
        /* Footer */
        .footer {
            background-color: #032541;
            padding: 20px 40px;
            text-align: center;
            color: rgba(255,255,255,0.7);
            font-size: 0.9rem;
            border-top: 1px solid rgba(1,180,228,0.2);
            margin-top: 40px;
        }
        
        .footer-author {
            color: #01b4e4;
            font-weight: 600;
        }
        
        /* Messages */
        .stSuccess, .stInfo, .stWarning {
            background-color: white;
            color: #032541;
            border-radius: 8px;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #0d253f;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #01b4e4;
            border-radius: 5px;
        }
        
        /* Ensure column gap is minimal */
        [data-testid="column"] {
            padding: 0 5px !important;
        }
        
        @media (max-width: 768px) {
            .hero-content {
                flex-direction: column;
                text-align: center;
            }
            
            .hero-stats {
                justify-content: center;
                margin-top: 20px;
            }
        }
        </style>
    """, unsafe_allow_html=True)


# ============================
# DATA LOADING
# ============================

@st.cache_data
def load_data():
    """Loads movie data"""
    return load_movies_data("movies.csv")


@st.cache_resource
def load_model():
    """Load ALS model from .pkl or from ZIP containing a .pkl"""
    
    
    zip_path = "trained_models/als_model_latest.zip"

  
    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                pkl_files = [f for f in z.namelist() if f.endswith('.pkl')]
                if len(pkl_files) != 1:
                    st.warning(f"⚠️ Expected exactly 1 .pkl inside ZIP, found {len(pkl_files)}")
                    return ALSRecommenderModel()
                pkl_name = pkl_files[0]

                with z.open(pkl_name) as f:
                    data = pickle.load(f)

             
                if isinstance(data, ALSRecommenderModel):
                    model = data
               
                elif isinstance(data, dict):
                    model = ALSRecommenderModel()
                    model.user_factors = data.get('user_factors')
                    model.movie_factors = data.get('movie_factors')
                    model.movie_to_index = data.get('movie_to_index', {})
                    model.index_to_movie = data.get('index_to_movie', {})
                    model.hyperparameters = data.get('hyperparameters', {})
                    model.weight = data.get('weight', model.hyperparameters.get('taubias', 0.0))
                else:
                    st.warning(" Unknown object inside ZIP pickle")
                    return ALSRecommenderModel()

            
            return model

        except Exception as e:
            st.warning(f" Error loading ALS model from ZIP: {e}")
            st.info("Fallback to basic recommendation methods.")
            return ALSRecommenderModel()

    else:
        st.warning(f" ALS model not found (.pkl or .zip)")
        st.info("Please place your trained model in the 'trained_models/' folder.")
        return ALSRecommenderModel()


def init_session_state():
    """Initializes session state"""
    if 'ratings_manager' not in st.session_state:
        st.session_state.ratings_manager = UserRatingsManager()
    if 'recommendation_type' not in st.session_state:
        st.session_state.recommendation_type = "hybrid"
    if 'n_recommendations' not in st.session_state:
        st.session_state.n_recommendations = 10
    if 'selected_genre' not in st.session_state:
        st.session_state.selected_genre = "All Genres"
    if 'rating_mode' not in st.session_state:
        st.session_state.rating_mode = None
    if 'page_context' not in st.session_state:
        st.session_state.page_context = "popular"
    if 'last_ratings_count' not in st.session_state:
        st.session_state.last_ratings_count = 0


# ============================
# UTILITY FUNCTIONS
# ============================

def extract_title_and_year(full_title: str) -> tuple:
    """Extracts movie title and year from format 'Title (Year)'"""
    match = re.match(r'^(.+?)\s*\((\d{4})\)$', full_title)
    if match:
        return match.group(1).strip(), match.group(2)
    return full_title, None


def generate_unique_key(base: str, movie_id: int, context: str = "") -> str:
    """Generates a unique key for Streamlit widgets"""
    page_ctx = st.session_state.page_context
    unique_string = f"{base}_{movie_id}_{context}_{page_ctx}"
    hash_key = hashlib.md5(unique_string.encode()).hexdigest()[:8]
    return f"{base}_{movie_id}_{hash_key}"


# ============================
# TMDB API FUNCTIONS
# ============================

@st.cache_data(ttl=3600)
def get_movie_data_from_tmdb(title: str, year: Optional[str] = None) -> Optional[Dict]:
    """Fetches movie data from TMDB API"""
    if not TMDB_API_KEY:
        return {
            'poster_url': "https://via.placeholder.com/500x750/032541/FFFFFF?text=No+API+Key",
            'tmdb_id': None,
            'tmdb_url': None
        }
    
    try:
        params = {
            'api_key': TMDB_API_KEY,
            'query': title,
            'page': 1
        }
        if year:
            params['year'] = year
        
        response = requests.get(TMDB_SEARCH_URL, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                movie = data['results'][0]
                poster_path = movie.get('poster_path')
                tmdb_id = movie.get('id')
                
                poster_url = f"{TMDB_IMAGE_BASE_URL}{poster_path}" if poster_path else None
                
                return {
                    'poster_url': poster_url or "https://via.placeholder.com/500x750/032541/FFFFFF?text=No+Poster",
                    'tmdb_id': tmdb_id,
                    'tmdb_url': f"{TMDB_MOVIE_BASE_URL}{tmdb_id}" if tmdb_id else None
                }
        
        return {
            'poster_url': "https://via.placeholder.com/500x750/032541/FFFFFF?text=No+Poster",
            'tmdb_id': None,
            'tmdb_url': None
        }
    
    except Exception:
        return {
            'poster_url': "https://via.placeholder.com/500x750/032541/FFFFFF?text=Error",
            'tmdb_id': None,
            'tmdb_url': None
        }


def get_movies_with_posters(movie_ids: List[int], df_movies: pd.DataFrame) -> List[Dict]:
    """Gets movies with TMDB posters and links"""
    movies = []
    for movie_id in movie_ids:
        movie_row = df_movies[df_movies['movieId'] == movie_id]
        if not movie_row.empty:
            movie_row = movie_row.iloc[0]
            full_title = movie_row['title']
            title, year = extract_title_and_year(full_title)
            
            tmdb_data = get_movie_data_from_tmdb(title, year)
            
            if tmdb_data:
                movies.append({
                    'movieId': movie_id,
                    'title': full_title,
                    'clean_title': title,
                    'genres': movie_row['genres'],
                    'year': year,
                    'poster_url': tmdb_data['poster_url'],
                    'tmdb_id': tmdb_data['tmdb_id'],
                    'tmdb_url': tmdb_data['tmdb_url']
                })
    
    return movies


def get_all_genres(df_movies: pd.DataFrame) -> List[str]:
    """Extracts all unique genres"""
    all_genres = set()
    for genres in df_movies['genres'].dropna():
        for genre in genres.split('|'):
            all_genres.add(genre.strip())
    return sorted(list(all_genres))


def filter_movies_by_genre(df_movies: pd.DataFrame, genre: str) -> pd.DataFrame:
    """Filters movies by genre"""
    if genre == "All Genres":
        return df_movies
    return df_movies[df_movies['genres'].str.contains(genre, na=False)]


# ============================
# RECOMMENDATION GENERATION
# ============================

def generate_recommendations(ratings: Dict[int, float], 
                            df_movies: pd.DataFrame,
                            als_model: ALSRecommenderModel,
                            rec_type: str,
                            n_recs: int) -> List[int]:
    """
    Generates recommendations based on user ratings
    """
    try:
        if rec_type == "collaborative":
            return get_collaborative_recommendations(ratings, df_movies, n_recs)
        
        elif rec_type == "content":
            if not ratings:
                return []
            best_id = max(ratings, key=ratings.get)
            return get_content_based_recommendations(best_id, df_movies, n_recs)
        
        else:  # hybrid - uses ALS model
            if not als_model.is_trained:
                return get_collaborative_recommendations(ratings, df_movies, n_recs)
            
            return hybrid_recommendation(ratings, als_model, df_movies, n_recs)
    
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return get_collaborative_recommendations(ratings, df_movies, n_recs)


# ============================
# DISPLAY FUNCTIONS
# ============================

def display_hero_with_stats():
    """Displays hero section with stats"""
    ratings = st.session_state.ratings_manager.get_all_ratings()
    avg_rating = sum(ratings.values()) / len(ratings) if ratings else 0
    max_rating = max(ratings.values()) if ratings else 0
    
    html = f"""
    <div class="hero-section">
        <div class="hero-content">
            <div class="hero-left">
                <h1 class="hero-title">Welcome.</h1>
                 <p>Movie Recommender System | Developed by <span class="footer-author">TIAO Eliasse</span>, AIMS Student</p>
                <p class="hero-subtitle">Millions of movies to discover. Explore now.</p>
            </div>
            <div class="hero-stats">
                <div class="hero-stat-item">
                    <span class="hero-stat-number">{len(ratings)}</span>
                    <span class="hero-stat-label">Rated Movies</span>
                </div>
                <div class="hero-stat-item">
                    <span class="hero-stat-number">{avg_rating:.1f}</span>
                    <span class="hero-stat-label">Avg Rating</span>
                </div>
                <div class="hero-stat-item">
                    <span class="hero-stat-number">{max_rating:.1f}</span>
                    <span class="hero-stat-label">Top Rating</span>
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown("""
        <div class="footer">
            <p>Movie Recommender System | Developed by <span class="footer-author">TIAO Eliasse</span>, AIMS Student</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)


def display_star_rating(movie_id: int, current_value: float = 0) -> Optional[float]:
    """Displays interactive star rating system"""
    # Initialize temp rating in session state if not exists
    temp_key = f"temp_rating_{movie_id}"
    if temp_key not in st.session_state:
        st.session_state[temp_key] = current_value
    
    # Display stars
    cols = st.columns(10)
    selected_rating = st.session_state[temp_key]
    
    for i, col in enumerate(cols):
        with col:
            star_value = (i + 1) * 0.5
            is_filled = star_value <= selected_rating
            star_icon = "★" if is_filled else "☆"
            
            star_key = generate_unique_key(f"star_{star_value}", movie_id, "click")
            if st.button(star_icon, key=star_key, help=f"Rate {star_value}/5.0"):
                st.session_state[temp_key] = star_value
                st.rerun()
    
    if selected_rating > 0:
        st.markdown(f'<div class="rating-value">{selected_rating}/5.0</div>', unsafe_allow_html=True)
    
    return st.session_state[temp_key]


def display_movie_card(movie: Dict, show_rating: bool = False):
    """Displays a single movie card"""
    poster_url = movie['poster_url']
    title = movie['title'].replace('"', '&quot;').replace("'", "&#39;")
    year = movie.get('year', '')
    movie_id = movie['movieId']
    tmdb_url = movie.get('tmdb_url')
    
    rating_html = ""
    if show_rating:
        rating = st.session_state.ratings_manager.get_rating(movie_id)
        if rating:
            rating_html = f'<div class="rating-badge">{rating}</div>'
    
    with st.container():
        st.markdown(f'<div class="movie-card-wrapper"><div class="movie-card"><div class="poster-wrapper"><img src="{poster_url}" class="poster-image" alt="{title}" loading="lazy">{rating_html}</div><div class="movie-info"><div class="movie-title">{title}</div><div class="movie-date">{year if year else ""}</div></div></div></div>', unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            rate_key = generate_unique_key("rate_btn", movie_id, "action")
            if st.button("Rate", key=rate_key, use_container_width=True):
                st.session_state.rating_mode = movie_id
                st.rerun()
        
        with col_b:
            if tmdb_url:
                st.link_button("Watch", tmdb_url, use_container_width=True)
        
        if st.session_state.rating_mode == movie_id:
            display_rating_modal(movie)


def display_rating_modal(movie: Dict):
    """Displays inline rating modal with star rating"""
    movie_id = movie['movieId']
    current_rating = st.session_state.ratings_manager.get_rating(movie_id)
    
    st.markdown('<div class="rating-modal">', unsafe_allow_html=True)
    
    if current_rating:
        st.markdown(f"### Your Rating")
        
        # Display current rating as stars (read-only)
        full_stars = int(current_rating)
        half_star = 1 if (current_rating - full_stars) >= 0.5 else 0
        empty_stars = 5 - full_stars - half_star
        
        star_display = "★" * full_stars
        if half_star:
            star_display += "★"  # Using filled star for half stars too for simplicity
        star_display += "☆" * empty_stars
        
        st.markdown(f'<div style="text-align: center; font-size: 2rem; color: #FFD700;">{star_display}</div>', 
                   unsafe_allow_html=True)
        st.markdown(f'<div style="text-align: center; color: #01b4e4; font-size: 1.2rem; font-weight: 600;">{current_rating}/5.0</div>', 
                   unsafe_allow_html=True)
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            mod_key = generate_unique_key("mod", movie_id, "modal")
            if st.button("Modify", key=mod_key, use_container_width=True):
                st.session_state.ratings_manager.remove_rating(movie_id)
                st.rerun()
        with col_b:
            del_key = generate_unique_key("del", movie_id, "modal")
            if st.button("Delete", key=del_key, use_container_width=True):
                st.session_state.ratings_manager.remove_rating(movie_id)
                st.session_state.rating_mode = None
                st.success("Rating deleted!")
                time.sleep(0.5)
                st.rerun()
        with col_c:
            close_key = generate_unique_key("close", movie_id, "modal")
            if st.button("Close", key=close_key, use_container_width=True):
                st.session_state.rating_mode = None
                st.rerun()
    else:
        st.markdown(f"### Rate: {movie['title']}")
        
        # Interactive star rating
        rating = display_star_rating(movie_id, current_value=3.0)
        
        col_a, col_b = st.columns(2)
        with col_a:
            sub_key = generate_unique_key("sub", movie_id, "modal")
            if st.button("Submit Rating", key=sub_key, use_container_width=True):
                if rating > 0:
                    st.session_state.ratings_manager.add_rating(movie_id, rating)
                    st.session_state.rating_mode = None
                    # Clean up temp rating
                    temp_key = f"temp_rating_{movie_id}"
                    if temp_key in st.session_state:
                        del st.session_state[temp_key]
                    st.success(f"✓ Rating of {rating}/5.0 saved!")
                    time.sleep(0.3)
                    st.rerun()
                else:
                    st.warning("Please select a rating!")
        with col_b:
            cancel_key = generate_unique_key("cancel", movie_id, "modal")
            if st.button("Cancel", key=cancel_key, use_container_width=True):
                st.session_state.rating_mode = None
                # Clean up temp rating
                temp_key = f"temp_rating_{movie_id}"
                if temp_key in st.session_state:
                    del st.session_state[temp_key]
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def display_movie_grid(movies: List[Dict], show_rating: bool = False):
    """Displays movies in a grid with minimal spacing"""
    if not movies:
        st.info("No movies to display.")
        return
    
    cols_per_row = 6
    for i in range(0, len(movies), cols_per_row):
        cols = st.columns(cols_per_row, gap="small")
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(movies):
                with col:
                    display_movie_card(movies[idx], show_rating)


# ============================
# MAIN APPLICATION
# ============================

def main():
    """Main application function"""
    
    load_custom_css()
    init_session_state()
    df_movies = load_data()
    als_model = load_model()
    
    all_genres = ["All Genres"] + get_all_genres(df_movies)
    
    # Display model status in sidebar
    with st.sidebar:
        st.title("⚙️ System Status")
        if als_model.is_trained:
            st.success("✓ ALS Model Active")
            with st.expander("📊 Model Info"):
                st.text(f"Movies: {len(als_model.movie_to_index)}")
                st.text(f"K: {als_model.hyperparameters.get('K', 'N/A')}")
                st.text(f"Weight: {als_model.weight}")
                st.text(f"Lambda: {als_model.hyperparameters.get('lambda_reg', 'N/A')}")
        else:
            st.info("ℹ️ Using Basic Methods")
    
    # Hero with Stats
    display_hero_with_stats()
    
    # Search Section
    st.markdown('<div class="search-section">', unsafe_allow_html=True)
    col_search, col_btn = st.columns([5, 1])
    with col_search:
        search_query = st.text_input(
            "Search",
            placeholder="Search for a movie...",
            key="main_search",
            label_visibility="collapsed"
        )
    with col_btn:
        search_button = st.button("Search", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Control Panel
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([2, 1.5, 1, 1])
    
    with col1:
        genre_filter = st.selectbox(
            "Filter by Genre", 
            all_genres,
            index=all_genres.index(st.session_state.selected_genre) if st.session_state.selected_genre in all_genres else 0,
            key="genre_select"
        )
        st.session_state.selected_genre = genre_filter
    
    with col2:
        rec_options = ["hybrid", "collaborative", "content"]
        rec_labels = {
            "hybrid": "Hybrid (ALS Model)",
            "collaborative": "Collaborative Filtering", 
            "content": "Content-Based"
        }
        rec_type = st.selectbox(
            "Recommendation Method",
            rec_options,
            format_func=lambda x: rec_labels[x],
            index=rec_options.index(st.session_state.recommendation_type),
            key="rec_type_select"
        )
        st.session_state.recommendation_type = rec_type
    
    with col3:
        results_options = [10, 15, 20, 30]
        n_recs = st.selectbox(
            "Number of Results", 
            results_options,
            index=results_options.index(st.session_state.n_recommendations),
            key="n_recs_select"
        )
        st.session_state.n_recommendations = n_recs
    
    with col4:
        if st.button("Reset All Ratings", use_container_width=True):
            st.session_state.ratings_manager.clear_ratings()
            st.session_state.rating_mode = None
            st.session_state.last_ratings_count = 0
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["🎬 Popular Movies", "⭐ Recommendations", "📋 My Ratings"])
    
    # TAB 1: POPULAR MOVIES
    with tab1:
        st.session_state.page_context = "popular"
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        
        if search_query or search_button:
            results = search_movies(search_query, df_movies, limit=30)
            if not results.empty:
                if genre_filter != "All Genres":
                    results = filter_movies_by_genre(results, genre_filter)
                
                st.markdown(f'<div class="section-header"><h2 class="section-title">Search Results ({len(results)})</h2></div>', 
                           unsafe_allow_html=True)
                
                with st.spinner("Loading movies..."):
                    movies = get_movies_with_posters(results['movieId'].tolist(), df_movies)
                
                display_movie_grid(movies)
            else:
                st.warning("No movies found.")
        else:
            filtered_df = filter_movies_by_genre(df_movies, genre_filter)
            popular = filtered_df.head(st.session_state.n_recommendations)
            
            st.markdown('<div class="section-header"><h2 class="section-title">Popular Movies</h2></div>', 
                       unsafe_allow_html=True)
            
            with st.spinner("Loading popular movies..."):
                movies = get_movies_with_posters(popular['movieId'].tolist(), df_movies)
            
            display_movie_grid(movies)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 2: RECOMMENDATIONS
    with tab2:
        st.session_state.page_context = "recommendations"
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        
        ratings = st.session_state.ratings_manager.get_all_ratings()
        current_ratings_count = len(ratings)
        
        if current_ratings_count < 1:
            st.info("📽️ Rate at least 1 movie to get personalized recommendations!")
            st.markdown("Go to the **Popular Movies** tab and rate some movies you like.")
        else:
            with st.spinner("🎯 Generating personalized recommendations..."):
                rec_type = st.session_state.recommendation_type
                n_recs = st.session_state.n_recommendations
                
                rec_ids = generate_recommendations(ratings, df_movies, als_model, rec_type, n_recs)
                
                if rec_ids:
                    rec_movies = get_movies_with_posters(rec_ids, df_movies)
                    
                    method_name = {
                        "hybrid": "Hybrid (ALS Model)",
                        "collaborative": "Collaborative Filtering",
                        "content": "Content-Based"
                    }[rec_type]
                    
                    st.success(f"✓ {len(rec_movies)} recommendations generated using **{method_name}**")
                    st.info(f"📊 Based on your {current_ratings_count} rated movie{'s' if current_ratings_count > 1 else ''}")
                    
                    st.markdown('<div class="section-header"><h2 class="section-title">Recommended For You</h2></div>', 
                               unsafe_allow_html=True)
                    
                    display_movie_grid(rec_movies)
                    
                    if als_model.is_trained and rec_type == "hybrid":
                        with st.expander("🔍 View recommendation scores (Debug)"):
                            user_ratings_dict = ratings
                            try:
                                all_predictions = als_model.predict_for_user(user_ratings_dict, rec_ids)
                                st.write("**Top 10 predictions:**")
                                for i, (movie_id, score) in enumerate(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)[:10], 1):
                                    movie_title = df_movies[df_movies['movieId'] == movie_id]['title'].values[0] if len(df_movies[df_movies['movieId'] == movie_id]) > 0 else "Unknown"
                                    st.text(f"{i:2d}. {movie_title:<50} | Score: {score:.4f}")
                            except Exception as e:
                                st.error(f"Could not fetch prediction scores: {e}")
                else:
                    st.warning("No recommendations could be generated. Try rating more movies!")
            
            st.session_state.last_ratings_count = current_ratings_count
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # TAB 3: MY RATINGS
    with tab3:
        st.session_state.page_context = "ratings"
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        
        ratings = st.session_state.ratings_manager.get_all_ratings()
        
        if not ratings:
            st.info("You haven't rated any movies yet.")
            st.markdown("Go to the **Popular Movies** tab to start rating!")
        else:
            sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
            rated_ids = [mid for mid, _ in sorted_ratings]
            
            with st.spinner("Loading your rated movies..."):
                rated_movies = get_movies_with_posters(rated_ids, df_movies)
            
            st.markdown(f'<div class="section-header"><h2 class="section-title">Your Rated Movies ({len(ratings)})</h2></div>', 
                       unsafe_allow_html=True)
            
            display_movie_grid(rated_movies, show_rating=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div class="footer">
            <p>Movie Recommender System | Developed by <span class="footer-author">TIAO Eliasse</span>, AIMS Student</p>
            <p>Powered by TMDB API & ALS Recommendation Engine (Weight=0.05)</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()