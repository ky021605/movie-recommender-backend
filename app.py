import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import random 
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime
import requests

# Download the lexicon (runs silently if already downloaded)
nltk.download('vader_lexicon', quiet=True)

# Initialize the analyzer globally
sia = SentimentIntensityAnalyzer()

# ==========================================
# PHASE 1: SERVER INITIALIZATION
# ==========================================
app = Flask(__name__)
CORS(app)

print("Loading AI Model and Data into server memory...")

# 1. Load the LightFM Matrix Brain
try:
    with open('models/hybrid_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Warning: hybrid_model.pkl not found. AI will need to be trained.")
    model = None

# 2. Load the TMDB Global Data & VADER NLP Scores safely
try:
    df_movies = pd.read_csv("data/tmdb_movies_production.csv").fillna("")
    
    # Safe dictionaries for fast API lookups
    title_map = dict(zip(df_movies['movie_id'], df_movies['title']))
    poster_map = dict(zip(df_movies['movie_id'], df_movies['poster_path']))
    
    # AI & Metadata maps (Using safe .get fallbacks in case columns are missing)
    mood_map = dict(zip(df_movies['movie_id'], df_movies['vader_sentiment'])) if 'vader_sentiment' in df_movies.columns else {}
    overview_map = dict(zip(df_movies['movie_id'], df_movies['overview'])) if 'overview' in df_movies.columns else {}
    genre_map = dict(zip(df_movies['movie_id'], df_movies['genres'])) if 'genres' in df_movies.columns else {}
    rating_map = dict(zip(df_movies['movie_id'], df_movies['vote_average'])) if 'vote_average' in df_movies.columns else {}

except FileNotFoundError:
    print("CRITICAL: tmdb_movies_production.csv not found!")
    df_movies = pd.DataFrame()
    title_map, poster_map, mood_map, overview_map, genre_map = {}, {}, {}, {}, {}


# ==========================================
# PHASE 2: DATABASES
# ==========================================

# User authentication database
USER_DATABASE = {
    "admin": {
        "password": "password123", 
        "matrix_id": 42, 
        "preferred_genres": ["Action", "Sci-Fi"],
        "display_name": "Admin Boss",
        "birthday": "2000-01-01",
        "bio": "I love movies with explosions.",
        "needs_onboarding": False # Admin already has data
    },
    "testuser1": {
        "password": "abc", 
        "matrix_id": 105, 
        "preferred_genres": [],
        "display_name": "Test User",
        "birthday": "",
        "bio": "",
        "needs_onboarding": True # This user will trigger the new screen!
    }
}

# Live interactive memory bank
# Format: { tmdb_id: {"likes": int, "dislikes": int, "comments": [ {username, text} ] } }
SOCIAL_DB = {}


# ==========================================
# PHASE 3: API ROUTES
# ==========================================

@app.route('/api/login', methods=['POST'])
def login():
    """Authenticates a user and connects them to their AI profile."""
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if username in USER_DATABASE and USER_DATABASE[username]['password'] == password:
        return jsonify({
            "success": True, 
            "matrix_id": USER_DATABASE[username]['matrix_id'], 
            "username": username,
            "needs_onboarding": USER_DATABASE[username].get("needs_onboarding", False)
        })
    else:
        return jsonify({"success": False, "error": "Invalid username or password"}), 401

@app.route('/api/register', methods=['POST'])
def register():
    """Creates a new user and routes them immediately into the Onboarding Matrix."""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"success": False, "error": "Username and password required"}), 400
        
    if username in USER_DATABASE:
        return jsonify({"success": False, "error": "Username already exists. Please login."}), 400
        
    # Generate a new unique Matrix ID based on the highest existing ID
    new_matrix_id = max([u['matrix_id'] for u in USER_DATABASE.values()]) + 1 if USER_DATABASE else 100
    
    # Save the brand new user to our system
    USER_DATABASE[username] = {
        "password": password,
        "matrix_id": new_matrix_id,
        "preferred_genres": [],
        "display_name": username,
        "birthday": "",
        "bio": "",
        "needs_onboarding": True # Forces them to calibrate their AI matrix
    }
    
    # Auto-login the user immediately after registration
    return jsonify({
        "success": True, 
        "matrix_id": new_matrix_id,
        "username": username,
        "needs_onboarding": True 
    })

@app.route('/api/interact', methods=['POST'])
def handle_interaction():
    """Saves a like, dislike, or a comment to the local server memory AND the ML Pipeline."""
    data = request.json
    tmdb_id = str(data.get('tmdb_id'))
    action = data.get('action') 
    username = data.get('username', 'Anonymous') # Capture who is clicking
    
    # Initialize the movie if it has no interactions yet
    if tmdb_id not in SOCIAL_DB:
        SOCIAL_DB[tmdb_id] = {"likes": 0, "dislikes": 0, "comments": []}
        
    if action == 'like':
        SOCIAL_DB[tmdb_id]['likes'] += 1
        
        # --- NEW: EXPORT REAL CLICKS FOR LIGHTFM ---
        # Get the user's matrix ID (default to 42 for the admin)
        matrix_id = USER_DATABASE.get(username, {}).get("matrix_id", 42)
        os.makedirs("data", exist_ok=True)
        
        # Append the interaction to a CSV file for the training script to read
        with open("data/real_likes.csv", "a") as f:
            f.write(f"{matrix_id},{tmdb_id},1.0\n")
        # -------------------------------------------
            
    elif action == 'dislike':
        SOCIAL_DB[tmdb_id]['dislikes'] += 1 
    elif action == 'comment':
        text = data.get('text', '')
        if text:
            SOCIAL_DB[tmdb_id]['comments'].append({"username": username, "text": text})
            
    return jsonify({"success": True, "data": SOCIAL_DB[tmdb_id]})


sia = SentimentIntensityAnalyzer()

@app.route('/api/recommend', methods=['GET'])
def get_live_recommendations():
    """Generates mathematically personalized recommendations using the LightFM Matrix model, 
    with explicit genre boosting and a live TMDB API fallback.
    """
    try:
        # 1. Capture the logged-in user's Matrix ID dynamically
        user_id = request.args.get('user_id', '42')
        top_n = int(request.args.get('top_n', 6))
        
        recommendations = []
        used_lightfm = False
        
        # 2. MACHINE LEARNING ENGINE: Compute matrix predictions if model & dataset are loaded
        if model is not None and not df_movies.empty:
            try:
                user_matrix_id = int(user_id)
                num_items = len(df_movies)
                
                # --- NEW: EXTRACT USER'S PROFILE PREFERENCES ---
                # Search the database to find the genres this specific user added to their profile
                preferred_genres = []
                for uname, udata in USER_DATABASE.items():
                    if udata.get("matrix_id") == user_matrix_id:
                        preferred_genres = udata.get("preferred_genres", [])
                        break
                # -----------------------------------------------

                # Run LightFM mathematical matrix factorization across all items for this user
                scores = model.predict(user_matrix_id, np.arange(num_items))
                
                # --- NEW: EXPLICIT GENRE BOOSTING (POST-PROCESSING) ---
                # If the user has profile genres, artificially boost those movies to the top
                if preferred_genres:
                    for i in range(num_items):
                        m_id = df_movies.iloc[i]['movie_id']
                        movie_genres = genre_map.get(m_id, "")
                        
                        # If any preferred genre is inside the movie's genre string, boost the AI score!
                        if any(pref.lower() in str(movie_genres).lower() for pref in preferred_genres):
                            scores[i] += 5.0  # Massive weight multiplier to guarantee priority
                # ------------------------------------------------------
                
                # --- EXPLORATION VS EXPLOITATION ---
                # 1. Grab a larger pool of highly recommended movies for this specific user (e.g., Top 30)
                pool_size = max(top_n, 30)
                if pool_size > num_items:
                    pool_size = num_items
    
                top_indices_pool = np.argsort(-scores)[:pool_size]

                # 2. Randomly shuffle and select 'top_n' movies from this elite pool
                selected_indices = np.random.choice(top_indices_pool, size=top_n, replace=False)

                # 3. Get the movie data for our selected shuffled list
                selected_rows = df_movies.iloc[selected_indices].to_dict('records')
                
                
                # Format predictions perfectly to feed your existing JavaScript renderMovies() grid
                for rank, row in enumerate(selected_rows):
                    m_id = int(row['movie_id'])
                    overview = row.get('overview', '')
                    mood_score = float(row.get('vader_sentiment', 0.0))
                    social_data = SOCIAL_DB.get(str(m_id), {"likes": 0, "dislikes": 0, "comments": []})
                    
                    # Extract the release year safely if present in your local CSV production file
                    release_year = ""
                    if 'release_date' in row and row['release_date']:
                        release_year = str(row['release_date'])[:4]
                    elif 'release_year' in row and row['release_year']:
                        release_year = str(row['release_year'])
                        
                    recommendations.append({
                        "rank": rank + 1,
                        "tmdb_id": m_id,
                        "title": row.get('title', "Unknown Title"),
                        "release_year": release_year,
                        "poster_path": row.get('poster_path', ""),
                        "overview": overview,
                        "genres": row.get('genres', "Cinema"),
                        "rating": round(float(row.get('vote_average', 0.0)), 1),
                        "likes": social_data["likes"],
                        "dislikes": social_data["dislikes"],
                        "comments": social_data["comments"],
                        "mood_score": mood_score
                    })
                
                used_lightfm = True
                print(f"🧠 [AI ENGINE] Successfully generated {len(recommendations)} personalized matrix matches for User {user_id}")
                
            except Exception as ml_error:
                print(f"⚠️ [AI WARNING] LightFM prediction out of bounds or failed: {str(ml_error)}. Routing to backup live array.")
        
        # 3. ROBUST FALLBACK ENGINE: If LightFM is bypassed or errors out, consult live TMDB Trending
        if not used_lightfm:
            print("🌐 [FALLBACK] Fetching real-time global trending feeds from TMDB API...")
            TMDB_TOKEN = os.getenv("TMDB_ACCESS_TOKEN")
            headers = {"Authorization": f"Bearer {TMDB_TOKEN}", "accept": "application/json"}
            
            random_page = random.randint(1, 10)
            url = f"https://api.themoviedb.org/3/trending/movie/week?language=en-US&page={random_page}"
            
            res = requests.get(url, headers=headers)
            if res.status_code != 200:
                return jsonify({"success": False, "error": "Failed to pull live feeds from TMDB API fallback"}), 500
                
            results = res.json().get('results', [])
            selected_movies = random.sample(results, min(len(results), top_n))
            
            genre_res = requests.get("https://api.themoviedb.org/3/genre/movie/list?language=en", headers=headers)
            genre_lookup = {g["id"]: g["name"] for g in genre_res.json().get("genres", [])} if genre_res.status_code == 200 else {}
            
            for rank, m in enumerate(selected_movies):
                m_id = m["id"]
                overview = m.get("overview", "")
                mood_score = sia.polarity_scores(overview)['compound'] if overview else 0.0
                social_data = SOCIAL_DB.get(str(m_id), {"likes": 0, "dislikes": 0, "comments": []})
                readable_genres = [genre_lookup.get(g_id, "Cinema") for g_id in m.get("genre_ids", [])]
                
                release_date = m.get("release_date", "")
                release_year = release_date[:4] if release_date else ""
                
                recommendations.append({
                    "rank": rank + 1,
                    "tmdb_id": m_id,
                    "title": m.get("title", "Unknown Title"),
                    "release_year": release_year,
                    "poster_path": m.get("poster_path", ""),
                    "overview": overview,
                    "genres": " | ".join(readable_genres) if readable_genres else "Cinema",
                    "rating": round(m.get("vote_average", 0.0), 1),
                    "likes": social_data["likes"],
                    "dislikes": social_data["dislikes"],
                    "comments": social_data["comments"],
                    "mood_score": mood_score
                })
                
        return jsonify({"success": True, "recommendations": recommendations})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/similar', methods=['GET'])
def get_similar_movies():
    """Returns movies with the closest emotional tone using NLP VADER scores."""
    try:
        target_id = int(request.args.get('tmdb_id'))
        top_n = int(request.args.get('top_n', 6))
        
        # 1. Try to get the mood from the map
        target_mood = mood_map.get(target_id) or mood_map.get(str(target_id))
        
       # 2. SELF-HEALING LOGIC: If missing, calculate it on the fly!
        if target_mood is None:
            print(f"DEBUG: Mood missing for {target_id}. Calculating on the fly...")
            
            # Fetch the plot summary from local database first
            overview = overview_map.get(target_id) or overview_map.get(str(target_id))
            
            # --- NEW LIVE FETCH LOGIC ---
            # If the movie isn't in our local database, ask TMDB for the plot directly!
            if not overview:
                TMDB_TOKEN = os.getenv("TMDB_ACCESS_TOKEN")
                url = f"https://api.themoviedb.org/3/movie/{target_id}?language=en-US"
                headers = {"Authorization": f"Bearer {TMDB_TOKEN}", "accept": "application/json"}
                
                res = requests.get(url, headers=headers)
                if res.status_code == 200:
                    overview = res.json().get("overview", "")
                    overview_map[target_id] = overview # Save it locally so it's faster next time!
            # -----------------------------
            
            if overview and str(overview).strip() != "":
                # Run VADER NLP Analysis
                target_mood = sia.polarity_scores(str(overview))['compound']
                
                # Save it into the dictionary so it doesn't have to calculate it again
                mood_map[target_id] = target_mood
                print(f"DEBUG: New mood score for {target_id} is {target_mood}")
            else:
                return jsonify({
                    "success": False, 
                    "error": f"Movie ID {target_id} has no plot summary. Cannot calculate Vibe."
                })

        # Calculate mathematical distance
        similarity_scores = []
        for m_id, mood in mood_map.items():
            # Ensure m_id is treated as an int for comparison
            if int(m_id) == target_id: 
                continue 
            
            mood_difference = abs(target_mood - mood)
            similarity_scores.append((m_id, mood_difference, mood))
            
        # Sort by closest match (smallest difference)
        similarity_scores.sort(key=lambda x: x[1])
        top_matches = similarity_scores[:top_n]
        
        recommended_movies = []
        for rank, (m_id, difference, mood) in enumerate(top_matches):
            m_id_str = str(m_id)
            social_data = SOCIAL_DB.get(m_id_str, {"likes": 0, "dislikes": 0, "comments": []})
            match_percentage = max(0, 100 - (difference * 50)) 

            recommended_movies.append({
                "rank": rank + 1,
                "tmdb_id": int(m_id),
                "title": title_map.get(int(m_id), "Unknown Title"),
                "poster_path": poster_map.get(int(m_id), ""),
                "likes": social_data["likes"],
                "dislikes": social_data["dislikes"],
                "comments": social_data["comments"],
                "mood_score": mood,
                "similarity_match": f"{round(match_percentage, 1)}%",
                "genres": genre_map.get(int(m_id), "Cinema"),
                "overview": overview_map.get(int(m_id), "No description available."),
                "rating": round(float(rating_map.get(int(m_id), 0.0)), 1)
            })
            
        # Grab the title so the frontend can display it in the "Because you liked..." header!
        target_title = title_map.get(int(target_id), "this movie")
        return jsonify({
            "success": True, 
            "target_id": target_id, 
            "target_title": target_title,
            "recommendations": recommended_movies
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}) # Also removed the 500 here for UI safety

@app.route('/api/genre/<genre_name>', methods=['GET'])
def get_movies_by_genre(genre_name):
    print(f"DEBUG: Searching for genre: {genre_name}") # This will print to your terminal if it's working
    filtered_df = df_movies[df_movies['genres'].str.contains(genre_name, case=False, na=False)]
    movies = filtered_df.head(10).to_dict('records')
    
    # Format the data exactly like your recommendation logic so renderMovies() accepts it
    results = []
    for rank, movie in enumerate(movies):
        m_id = int(movie['movie_id'])
        social_data = SOCIAL_DB.get(str(m_id), {"likes": 0, "dislikes": 0, "comments": []})
        results.append({
            "rank": rank + 1,
            "tmdb_id": m_id,
            "title": movie.get('title', "Unknown"),
            "poster_path": movie.get('poster_path', ""),
            "likes": social_data["likes"],
            "dislikes": social_data["dislikes"],
            "comments": social_data["comments"],
            "mood_score": movie.get('vader_sentiment', 0.0),
            "genres": movie.get('genres', "Cinema"),
            "overview": movie.get('overview', "No description."),
            "rating": round(float(movie.get('vote_average', 0.0)), 1)
        })
        
    return jsonify({"success": True, "recommendations": results})

@app.route('/api/movie-of-the-day', methods=['GET'])
def get_movie_of_the_day():
    try:
        # 1. Get today's date as a string (e.g., '2026-07-03')
        today_str = datetime.date.today().isoformat()
        
        # 2. Seed the random generator with today's date
        random.seed(today_str)
        
        # 3. Pick a movie (Ensure we have movies to pick from)
        movie_ids = list(title_map.keys())
        if not movie_ids:
            return jsonify({"success": False, "error": "No movies in database"}), 404
            
        # Optional: You can filter for highly rated movies only here
        featured_id = random.choice(movie_ids)
        
        # 4. RESET the seed to system time so it doesn't affect other random functions!
        random.seed() 
        
        movie_data = {
            "tmdb_id": featured_id,
            "title": title_map.get(featured_id, "Unknown Title"),
            "overview": overview_map.get(featured_id, "No description available."),
            "poster_path": poster_map.get(featured_id, ""),
            "genres": genre_map.get(featured_id, "Cinema"),
            "rating": rating_map.get(featured_id, 0.0)
        }
        
        return jsonify({"success": True, "movie": movie_data})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/trending-actors', methods=['GET'])
def get_trending_actors():
    """Fetches a randomized list of trending actors from TMDB, strictly filtering out adult content."""
    try:
        TMDB_TOKEN = os.getenv("TMDB_ACCESS_TOKEN")
        
        random_page = random.randint(1, 2)
        url = f"https://api.themoviedb.org/3/trending/person/week?language=en-US&page={random_page}"
        headers = {"Authorization": f"Bearer {TMDB_TOKEN}", "accept": "application/json"}
        
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            actors = res.json().get('results', [])
            
            # 1. NEW: Added 'not a.get("adult", False)' to completely ban adult actors
            valid_actors = [
                {"id": a["id"], "name": a["name"], "profile_path": a["profile_path"]} 
                for a in actors if a.get("profile_path") and not a.get("adult", False)
            ]
            
            if len(valid_actors) > 12:
                selected_actors = random.sample(valid_actors, 12)
            else:
                selected_actors = valid_actors
                
            return jsonify({"success": True, "actors": selected_actors})
            
        return jsonify({"success": False, "error": "TMDB API connection failed"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/actor/<int:actor_id>/movies', methods=['GET'])
def get_actor_movies(actor_id):
    """Fetches a specific actor's top movies, strictly filtering out adult films."""
    try:
        TMDB_TOKEN = os.getenv("TMDB_ACCESS_TOKEN")
        url = f"https://api.themoviedb.org/3/person/{actor_id}/movie_credits?language=en-US"
        headers = {"Authorization": f"Bearer {TMDB_TOKEN}", "accept": "application/json"}
        
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            movies = res.json().get('cast', [])
            
            sorted_movies = sorted(movies, key=lambda x: x.get('popularity', 0), reverse=True)[:15]
            
            recommendations = []
            for rank, m in enumerate(sorted_movies):
                
                # 2. NEW: Added 'not m.get("adult", False)' to ban any adult-rated movies from showing in the grid
                if m.get("poster_path") and not m.get("adult", False): 
                    
                    full_date = m.get("release_date", "")
                    release_year = full_date[:4] if full_date else "N/A"
                    
                    recommendations.append({
                        "rank": len(recommendations) + 1,
                        "tmdb_id": m["id"],
                        "title": m.get("title", "Unknown"),
                        "release_year": release_year,
                        "poster_path": m["poster_path"],
                        "genres": "Filmography",
                        "overview": m.get("overview", "No description available."),
                        "rating": round(m.get("vote_average", 0.0), 1),
                        "likes": 0, "dislikes": 0, "comments": [], "mood_score": 0
                    })
                    
                    # Stop once we have 12 clean, safe movies
                    if len(recommendations) == 12:
                        break
                        
            return jsonify({"success": True, "recommendations": recommendations})
            
        return jsonify({"success": False, "error": "Failed to fetch actor movies"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/profile/<username>', methods=['GET'])
def get_profile(username):
    """Fetches user profile details, personal info, and their history of liked movies."""
    user = USER_DATABASE.get(username)
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404
    
    liked_movies = []
    try:
        df_likes = pd.read_csv("data/real_likes.csv", names=["matrix_id", "tmdb_id", "weight"])
        user_likes = df_likes[df_likes["matrix_id"] == user["matrix_id"]]["tmdb_id"].tolist()
        
        for m_id in set(user_likes):
            if int(m_id) in title_map:
                liked_movies.append({
                    "tmdb_id": int(m_id),
                    "title": title_map.get(int(m_id), "Unknown"),
                    "poster_path": poster_map.get(int(m_id), ""),
                    "rating": float(rating_map.get(int(m_id), 0.0))
                })
    except FileNotFoundError:
        pass 
    
    liked_movies.sort(key=lambda x: x['rating'], reverse=True)
    
    return jsonify({
        "success": True, 
        "genres": user.get("preferred_genres", []),
        "liked_movies": liked_movies,
        "display_name": user.get("display_name", ""),
        "birthday": user.get("birthday", ""),
        "bio": user.get("bio", "")
    })

@app.route('/api/profile/update_details', methods=['POST'])
def update_profile_details():
    """Allows users to update their personal details."""
    data = request.json
    username = data.get('username')
    
    if username in USER_DATABASE:
        USER_DATABASE[username]['display_name'] = data.get('display_name', '')
        USER_DATABASE[username]['birthday'] = data.get('birthday', '')
        USER_DATABASE[username]['bio'] = data.get('bio', '')
        return jsonify({"success": True})
        
    return jsonify({"success": False, "error": "User not found"})


@app.route('/api/movie/<int:movie_id>', methods=['GET'])
def get_movie_details(movie_id):
    """Fetches full movie details including Director and Actors for the Details Page."""
    try:
        # 1. Try to find the movie in our local AI database first
        local_movie = df_movies[df_movies['movie_id'] == movie_id]
        
        if not local_movie.empty:
            row = local_movie.iloc[0]
            release_date = str(row.get('release_date', row.get('release_year', '')))
            
            movie_data = {
                "tmdb_id": movie_id,
                "title": row.get('title', 'Unknown Title'),
                "overview": row.get('overview', 'No description available.'),
                "poster_path": row.get('poster_path', ''),
                "genres": row.get('genres', 'Cinema'),
                "rating": round(float(row.get('vote_average', 0.0)), 1),
                "director": row.get('director', 'Unknown'),
                "actors": row.get('actors', 'Unknown'),
                "release_year": release_date[:4] if release_date else ""
            }
        else:
            # 2. LIVE TMDB FALLBACK: If they clicked a brand new trending movie not in our DB
            TMDB_TOKEN = os.getenv("TMDB_ACCESS_TOKEN")
            headers = {"Authorization": f"Bearer {TMDB_TOKEN}", "accept": "application/json"}
            
            res = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US", headers=headers)
            if res.status_code != 200:
                return jsonify({"success": False, "error": "Movie not found"}), 404
                
            m = res.json()
            
            # Ask TMDB for the specific Cast and Crew list
            cred_res = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/credits?language=en-US", headers=headers)
            director, actors = "Unknown", "Unknown"
            
            if cred_res.status_code == 200:
                c_data = cred_res.json()
                director = next((c["name"] for c in c_data.get("crew", []) if c["job"] == "Director"), "Unknown")
                actors_list = [a["name"] for a in c_data.get("cast", [])[:5]] # Get top 5 actors
                actors = "|".join(actors_list) if actors_list else "Unknown"
                
            genre_names = [g['name'] for g in m.get('genres', [])]
            
            movie_data = {
                "tmdb_id": movie_id,
                "title": m.get('title', 'Unknown Title'),
                "overview": m.get('overview', 'No description available.'),
                "poster_path": m.get('poster_path', ''),
                "genres": "|".join(genre_names),
                "rating": round(m.get('vote_average', 0.0), 1),
                "director": director,
                "actors": actors,
                "release_year": m.get('release_date', '')[:4] if m.get('release_date') else ""
            }

        # 3. Attach their Social interactions (Likes/Dislikes)
        social_data = SOCIAL_DB.get(str(movie_id), {"likes": 0, "dislikes": 0, "comments": []})
        movie_data["social"] = social_data

        return jsonify({"success": True, "movie": movie_data})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/onboarding/movies', methods=['GET'])
def get_onboarding_movies():
    """Fetches a highly-rated, diverse set of movies for new users to choose from."""
    try:
        # Grab the top 50 highest rated movies from our local AI database
        top_movies = df_movies[df_movies['vote_average'] > 7.0]
        
        # Randomly sample 18 of them so the grid looks fresh every time
        if not top_movies.empty:
            sample_size = min(18, len(top_movies))
            selected = top_movies.sample(sample_size).to_dict('records')
            
            results = []
            for m in selected:
                results.append({
                    "tmdb_id": int(m['movie_id']),
                    "title": m.get('title', 'Unknown'),
                    "poster_path": m.get('poster_path', '')
                })
            return jsonify({"success": True, "movies": results})
        return jsonify({"success": False, "error": "No movies found."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/onboarding/complete', methods=['POST'])
def complete_onboarding():
    """Saves the user's initial movie selections straight into the ML pipeline."""
    data = request.json
    username = data.get('username')
    selected_movies = data.get('selected_movies', []) # Array of TMDB IDs
    
    if username in USER_DATABASE:
        matrix_id = USER_DATABASE[username]['matrix_id']
        
        # 1. Flip their status so they never see the onboarding screen again
        USER_DATABASE[username]['needs_onboarding'] = False
        
        # 2. Inject their choices into the Matrix CSV file!
        os.makedirs("data", exist_ok=True)
        with open("data/real_likes.csv", "a") as f:
            for m_id in selected_movies:
                f.write(f"{matrix_id},{m_id},1.0\n")
                
                # Instantly add a like to the social database too
                if str(m_id) not in SOCIAL_DB:
                    SOCIAL_DB[str(m_id)] = {"likes": 1, "dislikes": 0, "comments": []}
                else:
                    SOCIAL_DB[str(m_id)]['likes'] += 1
                    
        return jsonify({"success": True})
        
    return jsonify({"success": False, "error": "User not found"})

@app.route('/api/top10', methods=['GET'])
def get_top_10():
    """Fetches the Top 10 most popular movies in Malaysia right now."""
    try:
        TMDB_TOKEN = os.getenv("TMDB_ACCESS_TOKEN")
        headers = {"Authorization": f"Bearer {TMDB_TOKEN}", "accept": "application/json"}
        
        # Notice the region=MY parameter! This makes it hyper-localized to Malaysia.
        url = "https://api.themoviedb.org/3/movie/popular?language=en-US&page=1&region=MY"
        
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            movies = res.json().get('results', [])[:10] # Slice exactly the first 10
            
            results = []
            for rank, m in enumerate(movies):
                results.append({
                    "rank": rank + 1,
                    "tmdb_id": m["id"],
                    "title": m.get("title", "Unknown"),
                    "poster_path": m.get("poster_path", "")
                })
            return jsonify({"success": True, "top10": results})
            
        return jsonify({"success": False, "error": "TMDB connection failed"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ==========================================
# PHASE 4: IGNITION
# ==========================================
if __name__ == '__main__':
    print("\n🚀 Starting the Recommendation Web Engine...")
    print("Test it in your browser at: http://127.0.0.1:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True)