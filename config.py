"""
Fichier de configuration pour l'application de recommandation de films
"""

# ============================
# CONFIGURATION API
# ============================

# TMDB API (The Movie Database)
# Obtenir une clé gratuite sur : https://www.themoviedb.org/settings/api
TMDB_API_KEY = "6dfcc050917bae29f25185975f23c0e9"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# OMDb API (Alternative)
# Obtenir une clé gratuite sur : http://www.omdbapi.com/apikey.aspx
OMDB_API_KEY = "YOUR_OMDB_API_KEY_HERE"
OMDB_BASE_URL = "http://www.omdbapi.com/"


# ============================
# CONFIGURATION FICHIERS
# ============================

# Chemins des fichiers de données
MOVIES_FILE = "movies.csv"
RATINGS_FILE = "ratings.csv"  # Si vous avez un fichier de ratings existant
MODEL_FILE = "als_model.pkl"  # Fichier du modèle ALS sauvegardé


# ============================
# CONFIGURATION MODÈLE ALS
# ============================

# Paramètres du modèle ALS
ALS_NUM_FACTORS = 50  # Nombre de facteurs latents
ALS_NUM_ITERATIONS = 30  # Nombre d'itérations
ALS_LAMBDA_REG = 0.01  # Paramètre de régularisation


# ============================
# CONFIGURATION RECOMMANDATIONS
# ============================

# Nombre par défaut de recommandations
DEFAULT_N_RECOMMENDATIONS = 10

# Nombre minimum de films à noter avant de pouvoir générer des recommandations
MIN_RATINGS_FOR_RECOMMENDATIONS = 3

# Poids pour la recommandation hybride
HYBRID_ALS_WEIGHT = 0.7
HYBRID_CONTENT_WEIGHT = 0.3


# ============================
# CONFIGURATION INTERFACE
# ============================

# Nombre de films à afficher par ligne dans la grille
MOVIES_PER_ROW = 5

# Nombre maximum de résultats de recherche
MAX_SEARCH_RESULTS = 20

# Limite de temps pour les requêtes API (en secondes)
API_TIMEOUT = 5

# Délai entre les requêtes API pour éviter le rate limiting (en secondes)
API_DELAY = 0.2


# ============================
# CONFIGURATION COULEURS
# ============================

# Couleur principale (Netflix rouge)
PRIMARY_COLOR = "#E50914"

# Couleur secondaire
SECONDARY_COLOR = "#831010"

# Couleur de fond
BACKGROUND_COLOR = "#141414"

# Couleur des cartes
CARD_COLOR = "#1f1f1f"


# ============================
# MESSAGES D'INTERFACE
# ============================

MESSAGES = {
    "welcome": "Bienvenue sur le système de recommandation de films !",
    "no_ratings": "Vous n'avez pas encore noté de films. Commencez par noter quelques films pour obtenir des recommandations personnalisées.",
    "min_ratings": f"Notez au moins {MIN_RATINGS_FOR_RECOMMENDATIONS} films pour obtenir des recommandations.",
    "loading_recommendations": "Génération de vos recommandations personnalisées...",
    "success_rating": "Note enregistrée avec succès !",
    "error_api": "Erreur lors de la récupération des données. Veuillez réessayer.",
}


# ============================
# CONFIGURATION AVANCÉE
# ============================

# Activer/désactiver le cache
USE_CACHE = True

# Activer/désactiver les logs détaillés
DEBUG_MODE = False

# Utiliser TMDB pour les posters
USE_TMDB = True

# Utiliser OMDb pour les posters (en fallback)
USE_OMDB = True
