# from flask import Flask, jsonify, request, render_template
# import pandas as pd
# from flask_cors import CORS
# import unicodedata
# import re
# import numpy as np
# from sentence_transformers import SentenceTransformer, util

# app = Flask(__name__)
# CORS(app)

# # Normaliser les textes 
# def normalize_text(s: str) -> str:
#     if not isinstance(s, str):
#         s = str(s)
#     s = s.strip()
#     s = unicodedata.normalize('NFD', s)
#     s = ''.join(ch for ch in s if not unicodedata.combining(ch))
#     s = s.lower()
#     s = re.sub(r'\s+', ' ', s)
#     return s

# # Chargement du CSV
# csv_file = 'df_final_plus1.csv'
# try:
#     df_events = pd.read_csv(csv_file)
# except FileNotFoundError:
#     print(f"ERREUR: Le fichier '{csv_file}' est introuvable.")
#     exit()

# required_cols = ['lat', 'lon', 'Category', 'City', 'Description', 'EventName']
# missing_cols = [col for col in required_cols if col not in df_events.columns]
# if missing_cols:
#     print(f"ERREUR: Colonnes manquantes dans le CSV: {missing_cols}")
#     exit()

# df_events['lat'] = pd.to_numeric(df_events['lat'], errors='coerce')
# df_events['lon'] = pd.to_numeric(df_events['lon'], errors='coerce')
# df_events.fillna('', inplace=True)

# # Normalisation des catégories
# df_events['_cat_norm'] = df_events['Category'].apply(normalize_text)

# # Charger le modèle LLM 
# model = SentenceTransformer('all-MiniLM-L6-v2')
# # Pré-calculer les embeddings de tous les événements
# event_embeddings = model.encode(
#     df_events['EventName'].fillna('') + ' ' + df_events['Description'].fillna(''),
#     convert_to_tensor=True
# )

# # ROUTE FRONT
# @app.route('/')
# def index():
#     return render_template('index.html')

# # API catégories
# @app.route('/api/categories', methods=['GET'])
# def get_categories():
#     categories = sorted(df_events['Category'].dropna().unique().tolist())
#     return jsonify(categories)

# # Filtre par catégorie
# def filter_by_category(df, interests_param):
#     if not interests_param:
#         return df
#     interests = [normalize_text(i) for i in interests_param.split(',') if i.strip()]
#     df_filtered = df[df['_cat_norm'].isin(interests)]
#     return df_filtered

# # SMART SEARCH : recherche sémantique
# @app.route('/api/smart-search', methods=['GET'])
# def smart_search():
#     interests_param = request.args.get('interests', '').strip()
#     query = request.args.get('q', '').strip()

#     df_filtered = filter_by_category(df_events, interests_param)

#     if query:
#         query_embedding = model.encode(query, convert_to_tensor=True)
#         filtered_embeddings = event_embeddings[df_filtered.index.tolist()]
#         cos_scores = util.cos_sim(query_embedding, filtered_embeddings)[0].cpu().numpy()
#         top_idx = np.argsort(-cos_scores)
#         df_filtered = df_filtered.iloc[top_idx]

#     return jsonify(df_filtered.to_dict('records'))

# # CITIES BY LLM
# @app.route('/api/cities-by-llm', methods=['GET'])
# def cities_by_llm():
#     interests_param = request.args.get('interests', '').strip()
#     query = request.args.get('q', '').strip()

#     df_filtered = filter_by_category(df_events, interests_param) if interests_param else df_events
#     df_filtered = df_filtered[df_filtered['City'].astype(str).str.strip() != '']

#     if query:
#         query_embedding = model.encode(query, convert_to_tensor=True)
#         filtered_embeddings = event_embeddings[df_filtered.index.tolist()]
#         cos_scores = util.cos_sim(query_embedding, filtered_embeddings)[0].cpu().numpy()
#         top_idx = np.argsort(-cos_scores)
#         df_filtered = df_filtered.iloc[top_idx]

#     city_counts_df = (
#         df_filtered.groupby('City', as_index=False)
#         .size()
#         .rename(columns={'size': 'count'})
#         .sort_values(by='count', ascending=False)
#     )

#     return jsonify(city_counts_df.to_dict('records'))

# # EVENTS BY CITY
# @app.route('/api/events-by-city', methods=['GET'])
# def events_by_city():
#     city = request.args.get('city', '').strip()
#     interests_param = request.args.get('interests', '').strip()
#     query = request.args.get('q', '').strip()

#     if not city:
#         return jsonify({"error": "Paramètre 'city' manquant"}), 400

#     df_filtered = df_events[df_events['City'].astype(str).str.lower().str.strip() == city.lower()]
#     df_filtered = filter_by_category(df_filtered, interests_param)

#     if query:
#         query_embedding = model.encode(query, convert_to_tensor=True)
#         filtered_embeddings = event_embeddings[df_filtered.index.tolist()]
#         cos_scores = util.cos_sim(query_embedding, filtered_embeddings)[0].cpu().numpy()
#         top_idx = np.argsort(-cos_scores)
#         df_filtered = df_filtered.iloc[top_idx]

#     return jsonify(df_filtered.to_dict('records'))

# # LANCEMENT DU SERVEUR
# if __name__ == '__main__':
#     print("Serveur démarré sur http://127.0.0.1:5000")
#     app.run(debug=True, port=5000)




from flask import Flask, jsonify, request, render_template
import pandas as pd
from flask_cors import CORS
import unicodedata
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util




app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------
# 🔧 NORMALISATION TEXTE
# ---------------------------------------------------------
def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r'\s+', ' ', s)
    return s

# ---------------------------------------------------------
# 📥 CHARGEMENT CSV
# ---------------------------------------------------------
csv_file = 'df_final_trad.csv'
try:
    df_events = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"ERREUR: Le fichier '{csv_file}' est introuvable.")
    exit()

# Vérification colonnes essentielles
required_cols = ['lat', 'lon', 'Category', 'City', 'Description', 'EventName']
missing = [c for c in required_cols if c not in df_events.columns]
if missing:
    print("ERREUR colonnes manquantes:", missing)
    exit()

# Nettoyage lat/lon
df_events['lat'] = pd.to_numeric(df_events['lat'], errors='coerce')
df_events['lon'] = pd.to_numeric(df_events['lon'], errors='coerce')

df_events.fillna('', inplace=True)

# ---------------------------------------------------------
# 📅 CONVERSION DES DATES
# ---------------------------------------------------------
if "DateTime_start" in df_events.columns:
    df_events["DateTime_start"] = pd.to_datetime(df_events["DateTime_start"], errors="coerce")
else:
    df_events["DateTime_start"] = pd.NaT

# ---------------------------------------------------------
# 🗂 NORMALISATION CATÉGORIES
# ---------------------------------------------------------
df_events["_cat_norm"] = df_events["Category"].apply(normalize_text)

# ---------------------------------------------------------
# 🤖 MODÈLE EMBEDDINGS
# ---------------------------------------------------------
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')



event_embeddings = model.encode(
    df_events['EventName'].fillna('') + ' ' + df_events['Description'].fillna(''),
    convert_to_tensor=True
)

# ---------------------------------------------------------
# 🧹 FILTRES
# ---------------------------------------------------------
def filter_by_category(df, interests_param):
    if not interests_param:
        return df
    interests = [normalize_text(i) for i in interests_param.split(',') if i.strip()]
    return df[df["_cat_norm"].isin(interests)]

def filter_by_date(df, start, end):
    if "DateTime_start" not in df.columns:
        return df

    if start:
        try:
            start = pd.to_datetime(start)
            df = df[df["DateTime_start"] >= start]
        except:
            pass

    if end:
        try:
            end = pd.to_datetime(end)
            df = df[df["DateTime_start"] <= end]
        except:
            pass

    return df

# ---------------------------------------------------------
# 🌐 ROUTE FRONT : page HTML principale
# ---------------------------------------------------------
@app.route('/')
def index():
    return render_template("index.html")


# ---------------------------------------------------------
# 📌 API : Liste des catégories
# ---------------------------------------------------------
@app.route('/api/categories')
def api_categories():
    categories = sorted(df_events["Category"].dropna().unique().tolist())
    return jsonify(categories)


# ---------------------------------------------------------
# 🔎 API : SMART SEARCH (recherche + filtres + tri)
# ---------------------------------------------------------
@app.route('/api/smart-search')
def smart_search():

    # ---- Récupération des paramètres ----
    interests = request.args.get("interests", "")
    query = request.args.get("q", "").strip()
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")
    sort_param = request.args.get("sort", "")   # <-- support tri par date

    # ---- Application filtres ----
    df_f = filter_by_category(df_events, interests)
    df_f = filter_by_date(df_f, start_date, end_date)

    # ---- Recherche vectorielle si mot-clé ----
    if query:
        q_emb = model.encode(query, convert_to_tensor=True)
        filt_emb = event_embeddings[df_f.index.tolist()]
        scores = util.cos_sim(q_emb, filt_emb)[0].cpu().numpy()
        df_f = df_f.iloc[np.argsort(-scores)]

    # ---- Tri chronologique (nouveau) ----
    if sort_param == "date" and "DateTime_start" in df_f.columns:
        df_f = df_f.sort_values("DateTime_start", ascending=True)

    return jsonify(df_f.to_dict("records"))


# ---------------------------------------------------------
# 🏙 API : Villes recommandées (ranking par pertinence + count)
# ---------------------------------------------------------
@app.route('/api/cities-by-llm')
def cities_by_llm():

    # ---- Récupération des paramètres ----
    interests = request.args.get("interests", "")
    query = request.args.get("q", "").strip()
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")

    # ---- Filtres ----
    df_f = filter_by_category(df_events, interests)
    df_f = filter_by_date(df_f, start_date, end_date)

    # Supprimer entrées sans ville
    df_f = df_f[df_f["City"].astype(str).str.strip() != ""]

    # ---- Recherche vectorielle ----
    if query:
        q_emb = model.encode(query, convert_to_tensor=True)
        filt_emb = event_embeddings[df_f.index.tolist()]
        scores = util.cos_sim(q_emb, filt_emb)[0].cpu().numpy()
        df_f = df_f.iloc[np.argsort(-scores)]

    # ---- Regrouper par ville + compter ----
    city_counts = (
        df_f.groupby("City", as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("count", ascending=False)
    )

    return jsonify(city_counts.to_dict("records"))


# ---------------------------------------------------------
# 📍 API : Événements par ville
# ---------------------------------------------------------
@app.route('/api/events-by-city')
def events_by_city():

    # ---- Récupération paramètres ----
    city = request.args.get("city", "").strip()
    interests = request.args.get("interests", "")
    query = request.args.get("q", "").strip()
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")
    sort_param = request.args.get("sort", "")

    if not city:
        return jsonify({"error": "Paramètre city manquant"}), 400

    # ---- Filtrage ----
    df_f = df_events[df_events["City"].astype(str).str.lower().str.strip() == city.lower()]
    df_f = filter_by_category(df_f, interests)
    df_f = filter_by_date(df_f, start_date, end_date)

    # ---- Recherche vectorielle ----
    if query:
        q_emb = model.encode(query, convert_to_tensor=True)
        filt_emb = event_embeddings[df_f.index.tolist()]
        scores = util.cos_sim(q_emb, filt_emb)[0].cpu().numpy()
        df_f = df_f.iloc[np.argsort(-scores)]

    # ---- Tri par date si demandé ----
    if sort_param == "date" and "DateTime_start" in df_f.columns:
        df_f = df_f.sort_values("DateTime_start", ascending=True)

    return jsonify(df_f.to_dict("records"))

# ---------------------------------------------------------
# 🚀 LANCEMENT
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Serveur OK ➜ http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
