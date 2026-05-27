from flask import Flask, render_template, request, jsonify
import re
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

DATA_PATH = "data/movies_preprocessed.csv"

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "from", "of", "with", "by", "about",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "it", "its", "as",
    "into", "through", "over", "under", "after", "before", "new", "old", "movie", "film", "series", "show"
}

SYNONYMS = {
    "crime": ["criminal", "drug", "police", "murder", "detective", "gang", "syndicate"],
    "love": ["romance", "romantic", "relationship", "marriage", "heart"],
    "war": ["battle", "soldier", "army", "military", "fight"],
    "school": ["student", "teacher", "college", "class"],
    "space": ["planet", "galaxy", "astronaut", "mission", "future"],
    "family": ["mother", "father", "son", "daughter", "home"],
    "magic": ["wizard", "witch", "spell", "fantasy"],
    "money": ["business", "cash", "rich", "bank"],
    "horror": ["ghost", "fear", "killer", "dead", "mystery"],
    "comedy": ["funny", "laugh", "comic", "friend"],
    "egypt": ["egyptian", "cairo", "alexandria", "arab"],
    "action": ["fight", "attack", "chase", "weapon", "danger"]
}


def simple_stem(word):
    endings = ["ing", "edly", "edly", "ed", "ies", "s"]
    for ending in endings:
        if len(word) > len(ending) + 3 and word.endswith(ending):
            if ending == "ies":
                return word[:-3] + "y"
            return word[:-len(ending)]
    return word


def clean_query(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = []
    for token in text.split():
        if token not in STOPWORDS and len(token) > 1:
            tokens.append(simple_stem(token))
    return " ".join(tokens)


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    expected = ["doc_id", "title", "type", "year", "rating", "original_abstract", "processed_text"]
    for col in expected:
        if col not in df.columns:
            df[col] = ""
    df["title"] = df["title"].fillna("Unknown Title").astype(str)
    df["type"] = df["type"].fillna("unknown").astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    df["original_abstract"] = df["original_abstract"].fillna("").astype(str)
    df["processed_text"] = df["processed_text"].fillna("").astype(str)
    df["search_text"] = (df["title"].astype(str) + " " + df["processed_text"].astype(str)).str.strip()
    df = df[df["search_text"].str.len() > 0].reset_index(drop=True)
    return df


movies_df = load_dataset()
vectorizer = TfidfVectorizer(max_features=18000, ngram_range=(1, 2), min_df=2)
tfidf_matrix = vectorizer.fit_transform(movies_df["search_text"])
feature_names = np.array(vectorizer.get_feature_names_out())


def expand_query(query, use_feedback=True):
    cleaned = clean_query(query)
    terms = cleaned.split()
    expanded = list(terms)
    for term in terms:
        expanded.extend(SYNONYMS.get(term, []))

    if use_feedback and cleaned:
        q_vec = vectorizer.transform([" ".join(expanded)])
        scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
        top_idx = scores.argsort()[::-1][:5]
        if len(top_idx) > 0 and scores[top_idx[0]] > 0:
            centroid = tfidf_matrix[top_idx].mean(axis=0)
            top_terms_idx = np.asarray(centroid).flatten().argsort()[::-1][:8]
            feedback_terms = [feature_names[i] for i in top_terms_idx if feature_names[i] not in expanded]
            expanded.extend(feedback_terms[:5])

    unique_terms = []
    for term in expanded:
        if term not in unique_terms and len(term) > 1:
            unique_terms.append(term)
    return cleaned, unique_terms


def apply_filters(df, type_filter="all", year_from=None, year_to=None, min_rating=0):
    filtered = df.copy()
    if type_filter and type_filter != "all":
        filtered = filtered[filtered["type"].str.lower() == type_filter.lower()]
    if year_from:
        filtered = filtered[filtered["year"].fillna(-1) >= int(year_from)]
    if year_to:
        filtered = filtered[filtered["year"].fillna(99999) <= int(year_to)]
    if min_rating:
        filtered = filtered[filtered["rating"] >= float(min_rating)]
    return filtered


def search_movies(query, type_filter="all", year_from=None, year_to=None, min_rating=0, top_k=30, expand=True):
    cleaned, expanded_terms = expand_query(query, expand)
    if not cleaned and not expanded_terms:
        return [], cleaned, expanded_terms

    final_query = " ".join(expanded_terms) if expanded_terms else cleaned
    q_vec = vectorizer.transform([final_query])
    tfidf_scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

    title_tokens = set(clean_query(query).split())
    title_boost = movies_df["title"].str.lower().apply(
        lambda title: sum(0.18 for token in title_tokens if token in title)
    ).to_numpy()
    rating_boost = (movies_df["rating"].fillna(0).to_numpy() / 10) * 0.05
    final_scores = tfidf_scores + title_boost + rating_boost

    filtered = apply_filters(movies_df, type_filter, year_from, year_to, min_rating)
    if filtered.empty:
        return [], cleaned, expanded_terms

    result_indices = filtered.index.to_numpy()
    ranked_indices = result_indices[np.argsort(final_scores[result_indices])[::-1]]
    ranked_indices = [i for i in ranked_indices if final_scores[i] > 0][:top_k]

    results = []
    for rank, idx in enumerate(ranked_indices, start=1):
        row = movies_df.loc[idx]
        results.append({
            "rank": rank,
            "doc_id": row["doc_id"],
            "title": row["title"],
            "type": row["type"],
            "year": "Unknown" if pd.isna(row["year"]) else int(row["year"]),
            "rating": round(float(row["rating"]), 2),
            "score": round(float(final_scores[idx]), 4),
            "abstract": row["original_abstract"][:420] + ("..." if len(row["original_abstract"]) > 420 else "")
        })
    return results, cleaned, expanded_terms


def make_clusters():
    cluster_df = movies_df.copy()
    cluster_df["year_num"] = cluster_df["year"].fillna(cluster_df["year"].median())
    features = cluster_df[["year_num", "rating"]].copy()
    scaler = StandardScaler()
    x = scaler.fit_transform(features)
    k = 4
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_df["cluster"] = model.fit_predict(x)
    summary = []
    for c in sorted(cluster_df["cluster"].unique()):
        part = cluster_df[cluster_df["cluster"] == c]
        sample_titles = part.sort_values("rating", ascending=False)["title"].head(5).tolist()
        summary.append({
            "cluster": int(c),
            "count": int(len(part)),
            "avg_rating": round(float(part["rating"].mean()), 2),
            "avg_year": round(float(part["year_num"].mean()), 0),
            "top_titles": sample_titles
        })
    return summary


def recommend_by_title(title, top_k=8):
    matches = movies_df[movies_df["title"].str.lower().str.contains(str(title).lower(), na=False)]
    if matches.empty:
        return []
    idx = matches.index[0]
    scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    ranked = scores.argsort()[::-1]
    recommendations = []
    for r in ranked:
        if r == idx or scores[r] <= 0:
            continue
        row = movies_df.loc[r]
        recommendations.append({
            "title": row["title"],
            "type": row["type"],
            "year": "Unknown" if pd.isna(row["year"]) else int(row["year"]),
            "rating": round(float(row["rating"]), 2),
            "similarity": round(float(scores[r]), 4),
            "abstract": row["original_abstract"][:250] + ("..." if len(row["original_abstract"]) > 250 else "")
        })
        if len(recommendations) == top_k:
            break
    return recommendations


@app.route("/")
def home():
    types = ["all"] + sorted(movies_df["type"].dropna().unique().tolist())
    stats = {
        "total": len(movies_df),
        "movies": int((movies_df["type"].str.lower() == "movie").sum()),
        "shows": int((movies_df["type"].str.lower() != "movie").sum()),
        "avg_rating": round(float(movies_df["rating"].mean()), 2),
        "min_year": int(movies_df["year"].dropna().min()),
        "max_year": int(movies_df["year"].dropna().max())
    }
    return render_template("index.html", types=types, stats=stats)


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "")
    type_filter = request.args.get("type", "all")
    year_from = request.args.get("year_from", "") or None
    year_to = request.args.get("year_to", "") or None
    min_rating = request.args.get("min_rating", 0)
    expand = request.args.get("expand", "true") == "true"
    results, cleaned, expanded_terms = search_movies(query, type_filter, year_from, year_to, min_rating, expand=expand)
    return jsonify({
        "query": query,
        "processed_query": cleaned,
        "expanded_terms": expanded_terms,
        "count": len(results),
        "results": results
    })


@app.route("/api/analytics")
def api_analytics():
    type_counts = movies_df["type"].value_counts().head(8).to_dict()
    rating_bins = pd.cut(movies_df["rating"], bins=[-0.1, 2, 4, 6, 8, 10], labels=["0-2", "2-4", "4-6", "6-8", "8-10"]).value_counts().sort_index().to_dict()
    yearly = movies_df.dropna(subset=["year"]).copy()
    yearly["year"] = yearly["year"].astype(int)
    yearly_counts = yearly.groupby("year").size().tail(20).to_dict()
    top_rated = movies_df[movies_df["rating"] > 0].sort_values("rating", ascending=False).head(10)
    return jsonify({
        "type_counts": type_counts,
        "rating_bins": {str(k): int(v) for k, v in rating_bins.items()},
        "yearly_counts": {str(k): int(v) for k, v in yearly_counts.items()},
        "top_rated": top_rated[["title", "type", "year", "rating"]].fillna("Unknown").to_dict("records"),
        "clusters": make_clusters()
    })


@app.route("/api/recommend")
def api_recommend():
    title = request.args.get("title", "")
    return jsonify({"title": title, "recommendations": recommend_by_title(title)})


@app.route("/api/network")
def api_network():
    top_terms = feature_names[np.asarray(tfidf_matrix.sum(axis=0)).flatten().argsort()[::-1][:20]].tolist()
    type_nodes = movies_df["type"].value_counts().head(8).to_dict()
    return jsonify({
        "nodes_count": int(len(movies_df) + len(type_nodes) + len(top_terms)),
        "edges_count": int(len(movies_df) * 2),
        "main_type_nodes": type_nodes,
        "important_text_terms": top_terms
    })


if __name__ == "__main__":
    app.run(debug=True)
