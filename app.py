import os
import pickle
import pandas as pd
import numpy as np
import json
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
    raw_df = pd.read_csv("data/tmdb_movies_production.csv").fillna("")
    
    # --- FYP STRICT SAFETY FILTER ---
    if 'adult' in raw_df.columns:
        raw_df = raw_df[(raw_df['adult'] == False) | (raw_df['adult'] == 'False') | (raw_df['adult'] == '0')]
        
    if 'genres' in raw_df.columns:
        banned_words = "Erotic|Adult|Porn|Softcore"
        raw_df = raw_df[~raw_df['genres'].str.contains(banned_words, case=False, na=False)]
        
    df_movies = raw_df
    # --------------------------------
    
    title_map = dict(zip(df_movies['movie_id'], df_movies['title']))
    poster_map = dict(zip(df_movies['movie_id'], df_movies['poster_path']))
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

USER_DATABASE = {
    "admin": {
        "password": "password123", 
        "matrix_id": 42, 
        "preferred_genres": ["Action", "Sci-Fi"],
        "display_name": "Admin Boss",
        "birthday": "2000-01-01",
        "bio": "I love movies with explosions.",
        "needs_onboarding": False 
    },
    "testuser1": {
        "password": "abc", 
        "matrix_id": 105, 
        "preferred_genres": [],
        "display_name": "Test User",
        "birthday": "",
        "bio": "",
        "needs_onboarding": True 
    }
}

SOCIAL_DB = {}

# --- NEW: ADMIN BLOCKLIST MEMORY ---
BANNED_MOVIES = set()
# Put the exact ID numbers of the bad actors here!
BANNED_ACTORS = {3194176}

try:
    with open("data/banned.json", "r") as f:
        b_data = json.load(f)
        BANNED_MOVIES = set(b_data.get("movies", []))
        BANNED_ACTORS = set(b_data.get("actors", []))
except FileNotFoundError:
    pass
# -----------------------------------

# ==========================================
# PHASE 3: API ROUTES
# ==========================================

@app.route('/api/login', methods=['POST'])
def login():
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
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"success": False, "error": "Username and password required"}), 400
        
    if username in USER_DATABASE:
        return jsonify({"success": False, "error": "Username already exists. Please login."}), 400
        
    new_matrix_id = max([u['matrix_id'] for u in USER_DATABASE.values()]) + 1 if USER_DATABASE else 100
    
    USER_DATABASE[username] = {
        "password": password,
        "matrix_id": new_matrix_id,
        "preferred_genres": [],
        "display_name": username,
        "birthday": "",
        "bio": "",
        "needs_onboarding": True 
    }
    
    return jsonify({
        "success": True, 
        "matrix_id": new_matrix_id,
        "username": username,
        "needs_onboarding": True 
    })

@app.route('/api/interact', methods=['POST'])
def handle_interaction():
    data = request.json
    tmdb_id = str(data.get('tmdb_id'))
    action = data.get('action') 
    username = data.get('username', 'Anonymous') 
    
    if tmdb_id not in SOCIAL_DB:
        SOCIAL_DB[tmdb_id] = {"likes": 0, "dislikes": 0, "comments": []}
        
    if action == 'like':
        SOCIAL_DB[tmdb_id]['likes'] += 1
        matrix_id = USER_DATABASE.get(username, {}).get("matrix_id", 42)
        os.makedirs("data", exist_ok=True)
        with open("data/real_likes.csv", "a") as f:
            f.write(f"{matrix_id},{tmdb_id},1.0\n")
            
    elif action == 'dislike':
        SOCIAL_DB[tmdb_id]['dislikes'] += 1 
    elif action == 'comment':
        text = data.get('text', '')
        if text:
            SOCIAL_DB[tmdb_id]['comments'].append({"username": username, "text": text})
            
    return jsonify({"success": True, "data": SOCIAL_DB[tmdb_id]})


# --- NEW: ADMIN MODERATION ENDPOINT ---
# --- NEW: ADMIN MODERATION ENDPOINT WITH CHAIN BAN ---
@app.route('/api/admin/ban', methods=['POST'])
def admin_ban():
    """Human-in-the-Loop endpoint to permanently erase dangerous content."""
    data = request.json
    
    if data.get('username') != 'admin':
        return jsonify({"success": False, "error": "Unauthorized"}), 403
        
    item_type = data.get('type')
    item_id = int(data.get('id'))
    
    if item_type == 'movie':
        BANNED_MOVIES.add(item_id)
        print(f"🛡️ ADMIN BAN: Movie {item_id} has been erased.")
        
    elif item_type == 'actor':
        BANNED_ACTORS.add(item_id)
        print(f"🛡️ ADMIN BAN: Actor {item_id} has been erased.")
        
        # --- NEW CHAIN BAN LOGIC ---
        # Instantly ask TMDB for every movie this actor has made and ban them too
        try:
            TMDB_TOKEN = os.getenv("TMDB_ACCESS_TOKEN")
            headers = {"Authorization": f"Bearer {TMDB_TOKEN}", "accept": "application/json"}
            url = f"https://api.themoviedb.org/3/person/{item_id}/movie_credits?language=en-US"
            
            res = requests.get(url, headers=headers)
            if res.status_code == 200:
                # Add all their movies to the blocklist
                for movie in res.json().get('cast', []):
                    BANNED_MOVIES.add(movie['id'])
                print(f"🛡️ CHAIN BAN: Erased all movies starring Actor {item_id}.")
        except Exception as e:
            print(f"Failed to run chain ban: {e}")
        # ---------------------------
        
    os.makedirs("data", exist_ok=True)
    with open("data/banned.json", "w") as f:
        json.dump({"movies": list(BANNED_MOVIES), "actors": list(BANNED_ACTORS)}, f)
        
    return jsonify({"success": True})



sia = SentimentIntensityAnalyzer()

@app.route('/api/recommend', methods=['GET'])
def get_live_recommendations():
    try:
        user_id = request.args.get('user_id', '42')
        top_n = int(request.args.get('top_n', 6))
        recommendations = []
        used_lightfm = False
        
        if model is not None and not df_movies.empty:
            try:
                user_matrix_id = int(user_id)
                num_items = len(df_movies)
                
                preferred_genres = []
                for uname, udata in USER_DATABASE.items():
                    if udata.get("matrix_id") == user_matrix_id:
                        preferred_genres = udata.get("preferred_genres", [])
                        break

                scores = model.predict(user_matrix_id, np.arange(num_items))
                
                if preferred_genres:
                    for i in range(num_items):
                        m_id = df_movies.iloc[i]['movie_id']
                        movie_genres = genre_map.get(m_id, "")
                        if any(pref.lower() in str(movie_genres).lower() for pref in preferred_genres):
                            scores[i] += 5.0  
                
                pool_size = max(top_n * 2, 30)
                if pool_size > num_items:
                    pool_size = num_items
    
                top_indices_pool = np.argsort(-scores)[:pool_size]
                selected_indices = np.random.choice(top_indices_pool, size=pool_size, replace=False)
                selected_rows = df_movies.iloc[selected_indices].to_dict('records')
                
                for row in selected_rows:
                    m_id = int(row['movie_id'])
                    
                    # FYP SAFETY: Skip banned movies
                    if m_id in BANNED_MOVIES:
                        continue
                        
                    overview = row.get('overview', '')
                    mood_score = float(row.get('vader_sentiment', 0.0))
                    social_data = SOCIAL_DB.get(str(m_id), {"likes": 0, "dislikes": 0, "comments": []})
                    
                    release_year = ""
                    if 'release_date' in row and row['release_date']:
                        release_year = str(row['release_date'])[:4]
                    elif 'release_year' in row and row['release_year']:
                        release_year = str(row['release_year'])
                        
                    recommendations.append({
                        "rank": len(recommendations) + 1,
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
                    
                    if len(recommendations) == top_n:
                        break
                
                used_lightfm = True
                
            except Exception as ml_error:
                print(f"⚠️ [AI WARNING] {str(ml_error)}. Routing to backup.")
        
        if not used_lightfm:
            TMDB_TOKEN = os.getenv("TMDB_ACCESS_TOKEN")
            headers = {"Authorization": f"Bearer {TMDB_TOKEN}", "accept": "application/json"}
            
            random_page = random.randint(1, 10)
            url = f"https://api.themoviedb.org/3/trending/movie/week?language=en-US&page={random_page}"
            
            res = requests.get(url, headers=headers)
            if res.status_code != 200:
                return jsonify({"success": False, "error": "Failed API"}), 500
                
            results = res.json().get('results', [])
            
            safe_results = [m for m in results if not m.get("adult", False) and m.get("id") not in BANNED_MOVIES]
            selected_movies = random.sample(safe_results, min(len(safe_results), top_n))
            
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
    try:
        target_id = int(request.args.get('tmdb_id'))
        top_n = int(request.args.get('top_n', 6))
        
        target_mood = mood_map.get(target_id) or mood_map.get(str(target_id))
        
        if target_mood is None:
            overview = overview_map.get(target_id) or overview_map.get(str(target_id))
            if not overview:
                TMDB_TOKEN = os.getenv("TMDB_ACCESS_TOKEN")
                url = f"https://api.themoviedb.org/3/movie/{target_id}?language=en-US"
                headers = {"Authorization": f"Bearer {TMDB_TOKEN}", "accept": "application/json"}
                res = requests.get(url, headers=headers)
                if res.status_code == 200:
                    overview = res.json().get("overview", "")
                    overview_map[target_id] = overview 
            
            if overview and str(overview).strip() != "":
                target_mood = sia.polarity_scores(str(overview))['compound']
                mood_map[target_id] = target_mood
            else:
                return jsonify({"success": False, "error": f"Movie ID {target_id} has no plot summary. Cannot calculate Vibe."})

        similarity_scores = []
        for m_id, mood in mood_map.items():
            if int(m_id) == target_id or int(m_id) in BANNED_MOVIES: 
                continue 
            
            mood_difference = abs(target_mood - mood)
            similarity_scores.append((m_id, mood_difference, mood))
            
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
            
        target_title = title_map.get(int(target_id), "this movie")
        return jsonify({"success": True, "target_id": target_id, "target_title": target_title, "recommendations": recommended_movies})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}) 

@app.route('/api/genre/<genre_name>', methods=['GET'])
def get_movies_by_genre(genre_name):
    filtered_df = df_movies[df_movies['genres'].str.contains(genre_name, case=False, na=False)]
    
    # Filter out banned movies before picking the top 10
    safe_movies = filtered_df[~filtered_df['movie_id'].isin(BANNED_MOVIES)]
    movies = safe_movies.head(10).to_dict('records')
    
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
        today_str = datetime.date.today().isoformat()
        random.seed(today_str)
        
        movie_ids = [m for m in list(title_map.keys()) if m not in BANNED_MOVIES]
        if not movie_ids:
            return jsonify({"success": False, "error": "No movies in database"}), 404
            
        featured_id = random.choice(movie_ids)
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
    try:
        TMDB_TOKEN = os.getenv("TMDB_ACCESS_TOKEN")
        
        random_page = random.randint(1, 3)
        url = f"https://api.themoviedb.org/3/trending/person/week?language=en-US&page={random_page}"
        headers = {"Authorization": f"Bearer {TMDB_TOKEN}", "accept": "application/json"}
        
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            actors = res.json().get('results', [])
            
            bad_words = ["erotic", "porn", "softcore", "sex ", "lust", "seduce", "sensual", "desire", "vivamax", "nude", "nudity", "steamy", "affair", "scandal"]
            valid_actors = []
            
            for a in actors:
                # FYP SAFETY: Check the Admin Blocklist
                if a["id"] in BANNED_ACTORS:
                    continue
                    
                if not a.get("profile_path") or a.get("adult", False):
                    continue
                    
                is_safe = True
                mainstream_score = 0 
                
                for work in a.get("known_for", []):
                    text = str(work.get("overview", "")) + " " + str(work.get("title", "")) + " " + str(work.get("name", ""))
                    if any(bad in text.lower() for bad in bad_words):
                        is_safe = False
                        break 
                        
                    if work.get("vote_count", 0) > 150:
                        mainstream_score += 1
                        
                if is_safe and mainstream_score > 0:
                    valid_actors.append({"id": a["id"], "name": a["name"], "profile_path": a["profile_path"]})

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
    try:
        TMDB_TOKEN = os.getenv("TMDB_ACCESS_TOKEN")
        url = f"https://api.themoviedb.org/3/person/{actor_id}/movie_credits?language=en-US"
        headers = {"Authorization": f"Bearer {TMDB_TOKEN}", "accept": "application/json"}
        
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            movies = res.json().get('cast', [])
            sorted_movies = sorted(movies, key=lambda x: x.get('popularity', 0), reverse=True)
            
            bad_words = ["erotic", "porn", "softcore", "sex ", "lust", "seduce", "sensual", "desire", "vivamax", "nude", "nudity", "steamy", "affair", "scandal"]
            recommendations = []
            
            for m in sorted_movies:
                # FYP SAFETY: Check if admin banned this movie
                if m["id"] in BANNED_MOVIES:
                    continue
                    
                if m.get("poster_path") and not m.get("adult", False) and m.get("vote_count", 0) > 50: 
                    
                    text = str(m.get("overview", "")) + " " + str(m.get("title", ""))
                    if any(bad in text.lower() for bad in bad_words):
                        continue 
                        
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
                    
                    if len(recommendations) == 12:
                        break
                        
            return jsonify({"success": True, "recommendations": recommendations})
            
        return jsonify({"success": False, "error": "Failed to fetch actor movies"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/profile/<username>', methods=['GET'])
def get_profile(username):
    user = USER_DATABASE.get(username)
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404
    
    liked_movies = []
    try:
        df_likes = pd.read_csv("data/real_likes.csv", names=["matrix_id", "tmdb_id", "weight"])
        user_likes = df_likes[df_likes["matrix_id"] == user["matrix_id"]]["tmdb_id"].tolist()
        
        for m_id in set(user_likes):
            if int(m_id) in title_map and int(m_id) not in BANNED_MOVIES:
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
    # FYP SAFETY: Block detail requests for banned movies
    if movie_id in BANNED_MOVIES:
        return jsonify({"success": False, "error": "Classified Information"}), 404
        
    try:
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
            TMDB_TOKEN = os.getenv("TMDB_ACCESS_TOKEN")
            headers = {"Authorization": f"Bearer {TMDB_TOKEN}", "accept": "application/json"}
            
            res = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US", headers=headers)
            if res.status_code != 200:
                return jsonify({"success": False, "error": "Movie not found"}), 404
                
            m = res.json()
            
            cred_res = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/credits?language=en-US", headers=headers)
            director, actors = "Unknown", "Unknown"
            
            if cred_res.status_code == 200:
                c_data = cred_res.json()
                director = next((c["name"] for c in c_data.get("crew", []) if c["job"] == "Director"), "Unknown")
                actors_list = [a["name"] for a in c_data.get("cast", [])[:5]] 
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

        social_data = SOCIAL_DB.get(str(movie_id), {"likes": 0, "dislikes": 0, "comments": []})
        movie_data["social"] = social_data

        return jsonify({"success": True, "movie": movie_data})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/onboarding/movies', methods=['GET'])
def get_onboarding_movies():
    try:
        # Filter out banned movies
        safe_df = df_movies[~df_movies['movie_id'].isin(BANNED_MOVIES)]
        top_movies = safe_df[safe_df['vote_average'] > 7.0]
        
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
    data = request.json
    username = data.get('username')
    selected_movies = data.get('selected_movies', []) 
    
    if username in USER_DATABASE:
        matrix_id = USER_DATABASE[username]['matrix_id']
        USER_DATABASE[username]['needs_onboarding'] = False
        
        os.makedirs("data", exist_ok=True)
        with open("data/real_likes.csv", "a") as f:
            for m_id in selected_movies:
                f.write(f"{matrix_id},{m_id},1.0\n")
                
                if str(m_id) not in SOCIAL_DB:
                    SOCIAL_DB[str(m_id)] = {"likes": 1, "dislikes": 0, "comments": []}
                else:
                    SOCIAL_DB[str(m_id)]['likes'] += 1
                    
        return jsonify({"success": True})
        
    return jsonify({"success": False, "error": "User not found"})

@app.route('/api/top10', methods=['GET'])
def get_top_10():
    try:
        TMDB_TOKEN = os.getenv("TMDB_ACCESS_TOKEN")
        headers = {"Authorization": f"Bearer {TMDB_TOKEN}", "accept": "application/json"}
        
        url = "https://api.themoviedb.org/3/movie/popular?language=en-US&page=1&region=MY"
        
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            raw_movies = res.json().get('results', [])
            
            # FYP SAFETY: Remove adult content AND admin banned content BEFORE slicing top 10
            safe_movies = [m for m in raw_movies if not m.get("adult", False) and m.get("id") not in BANNED_MOVIES]
            movies = safe_movies[:10] 
            
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