Movies Search Engine - Data Mining Project
==========================================

This project uses the processed movies dataset from Phase 1 and builds a complete search engine for Phase 2 / Data Mining.

Main Features:
- Flask backend
- Clean web dashboard
- TF-IDF search ranking
- Query preprocessing
- Query expansion using synonyms and relevance feedback
- Filters by type, year, and rating
- Movie recommendation using cosine similarity
- Analytics charts
- K-Means clustering summary
- Network-style summary of movie/type/text-term relationships

How to run on Windows:
1. Extract the ZIP file.
2. Open the folder: MoviesSearchEngineReady
3. Double click: run_backend.bat
4. Open this link in your browser:
   http://127.0.0.1:5000

How to run manually:
1. Open CMD inside MoviesSearchEngineReady/backend
2. Run:
   python -m pip install -r requirements.txt
   python app.py
3. Open:
   http://127.0.0.1:5000

Dataset location:
backend/data/movies_preprocessed.csv

Important note for discussion:
The dataset is already cleaned and preprocessed from Phase 1. This phase focuses on data mining and information retrieval techniques such as TF-IDF ranking, query expansion, clustering, recommendations, and analytics.
