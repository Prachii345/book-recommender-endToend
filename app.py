import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import html
import random
from sklearn.metrics.pairwise import cosine_similarity



DATA_DIR  = "data"
MODEL_DIR = "models"

st.set_page_config(
    page_title="NextRead AI — Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for High-End Modern UI ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hero Section */
.hero-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
    border-radius: 20px;
    padding: 2.5rem 2.8rem;
    color: #ffffff;
    margin-bottom: 2rem;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.12);
    position: relative;
    overflow: hidden;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 5px 14px;
    border-radius: 30px;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.4);
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0 0 0.6rem 0;
    letter-spacing: -0.8px;
    color: #ffffff !important;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #cbd5e1;
    margin-bottom: 1.5rem;
    max-width: 800px;
    line-height: 1.6;
}

.hero-stats-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}

.stat-chip {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    padding: 8px 16px;
    border-radius: 40px;
    font-size: 0.85rem;
    color: #f1f5f9;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}

/* Book Card Grid */
.book-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(140, 140, 160, 0.22);
    border-radius: 16px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.07);
    transition: transform 0.28s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.28s ease, border-color 0.28s ease;
}

.book-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 16px 36px rgba(99, 102, 241, 0.18);
    border-color: rgba(99, 102, 241, 0.6);
}

.cover-wrapper {
    position: relative;
    width: 100%;
    height: 280px;
    background: #0f172a;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}

.book-cover {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.35s ease;
}

.book-card:hover .book-cover {
    transform: scale(1.05);
}

.rank-badge {
    position: absolute;
    top: 12px;
    left: 12px;
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(8px);
    color: #f8fafc;
    font-size: 0.75rem;
    font-weight: 800;
    padding: 4px 10px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.score-badge {
    position: absolute;
    top: 12px;
    right: 12px;
    color: #ffffff;
    font-size: 0.75rem;
    font-weight: 800;
    padding: 4px 12px;
    border-radius: 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    letter-spacing: 0.3px;
}

.card-details {
    padding: 1.2rem;
    display: flex;
    flex-direction: column;
    flex-grow: 1;
}

.book-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin: 0 0 0.35rem 0;
    line-height: 1.35;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    height: 2.8rem;
    color: inherit;
}

.book-author {
    font-size: 0.86rem;
    color: #6366f1;
    font-weight: 600;
    margin: 0 0 0.8rem 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.meta-row {
    display: flex;
    gap: 6px;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}

.pill-meta {
    font-size: 0.72rem;
    padding: 3px 9px;
    border-radius: 6px;
    background: rgba(128, 128, 128, 0.12);
    color: #94a3b8;
    font-weight: 500;
}

/* Score Breakdown Component */
.breakdown-box {
    margin-top: auto;
    padding-top: 0.8rem;
    border-top: 1px dashed rgba(140, 140, 160, 0.25);
}

.meter-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: #94a3b8;
    margin-bottom: 2px;
    font-weight: 500;
}

.meter-track {
    width: 100%;
    height: 5px;
    background: rgba(128, 128, 128, 0.14);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 6px;
}

.fill-svd {
    background: linear-gradient(90deg, #6366f1, #818cf8);
    height: 100%;
    border-radius: 4px;
}

.fill-content {
    background: linear-gradient(90deg, #06b6d4, #22d3ee);
    height: 100%;
    border-radius: 4px;
}

.fill-graph {
    background: linear-gradient(90deg, #f59e0b, #fbbf24);
    height: 100%;
    border-radius: 4px;
}

/* Preset & Button styles */
div.stButton > button:first-child {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
}

/* Info banner */
.query-banner {
    background: rgba(99, 102, 241, 0.08);
    border-left: 4px solid #6366f1;
    padding: 1rem 1.4rem;
    border-radius: 0 12px 12px 0;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load all models ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading book catalog and recommendation engines...")
def load_models():
    df       = pd.read_csv(os.path.join(DATA_DIR, "ratings_clean.csv"))
    books_df = pd.read_csv(os.path.join(DATA_DIR, "books_clean.csv"))
    books_df["content"] = books_df["content"].fillna("")

    with open(os.path.join(MODEL_DIR, "svd_model.pkl"), "rb") as f:
        svd_data = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "tfidf_matrix.pkl"), "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "title_to_idx.pkl"), "rb") as f:
        title_to_idx = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "book_graph.pkl"), "rb") as f:
        G = pickle.load(f)

    return df, books_df, svd_data, tfidf_matrix, title_to_idx, G


# ── Model helpers ───────────────────────────────────────────────
def get_content_scores(book_title, books_df, tfidf_matrix, title_to_idx, n=30):
    title_lower = book_title.lower()
    match = None
    for t in title_to_idx:
        if title_lower in t or t in title_lower:
            match = t
            break
    if match is None:
        return {}
    idx        = title_to_idx[match]
    query_vec  = tfidf_matrix[idx]
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idxs   = np.argsort(sim_scores)[::-1][1:n+1]
    return {books_df.iloc[i]["isbn"]: float(sim_scores[i]) for i in top_idxs}


def get_svd_scores(user_id, df, svd_data, n=30):
    user_enc     = svd_data["user_enc"]
    book_enc     = svd_data["book_enc"]
    book_dec     = svd_data["book_dec"]
    user_factors = svd_data["user_factors"]
    item_factors = svd_data["item_factors"]

    if user_id not in user_enc:
        return {}

    u_idx  = user_enc[user_id]
    u_vec  = user_factors[u_idx]
    scores = item_factors.T.dot(u_vec)

    rated_isbns = set(df[df["user_id"] == user_id]["isbn"].tolist())
    rated_idxs  = {book_enc[i] for i in rated_isbns if i in book_enc}
    ranked      = np.argsort(scores)[::-1]
    top_idxs    = [i for i in ranked if i not in rated_idxs][:n]

    top_scores = np.array([scores[i] for i in top_idxs])
    if len(top_scores) > 0 and top_scores.max() != top_scores.min():
        top_scores = (top_scores - top_scores.min()) / (top_scores.max() - top_scores.min())

    return {book_dec[i]: float(top_scores[j]) for j, i in enumerate(top_idxs)}


def get_graph_scores(start_isbn, G, n=30):
    if start_isbn not in G:
        return {}
    visited  = set()
    result   = {}
    frontier = list(G.neighbors(start_isbn))
    depth    = 1.0

    while frontier and len(result) < n:
        next_frontier = []
        for node in frontier:
            if node not in visited and node != start_isbn:
                visited.add(node)
                weights = [G[node][nb]["weight"] for nb in G.neighbors(node) if nb in visited or nb == start_isbn]
                score   = (np.mean(weights) if weights else 0.1) / depth
                result[node] = float(score)
                next_frontier.extend([nb for nb in G.neighbors(node) if nb not in visited])
        frontier = next_frontier
        depth   += 1.0

    if result:
        vals = np.array(list(result.values()))
        if vals.max() != vals.min():
            vals = (vals - vals.min()) / (vals.max() - vals.min())
        result = dict(zip(result.keys(), vals.tolist()))
    return result


def hybrid_recommend(user_id, liked_book_title, df, books_df,
                     svd_data, tfidf_matrix, title_to_idx, G,
                     alpha=0.4, beta=0.4, gamma=0.2, n=12):
    title_lower = liked_book_title.lower()
    liked_isbn  = None
    for t, idx in title_to_idx.items():
        if title_lower in t or t in title_lower:
            liked_isbn = books_df.iloc[idx]["isbn"]
            break

    svd_scores     = get_svd_scores(user_id, df, svd_data, n=40)
    content_scores = get_content_scores(liked_book_title, books_df, tfidf_matrix, title_to_idx, n=40)
    graph_scores   = get_graph_scores(liked_isbn, G, n=40) if liked_isbn else {}

    all_isbns = set(svd_scores) | set(content_scores) | set(graph_scores)

    hybrid_scores = {}
    for isbn in all_isbns:
        score = (alpha * svd_scores.get(isbn, 0) +
                 beta  * content_scores.get(isbn, 0) +
                 gamma * graph_scores.get(isbn, 0))
        hybrid_scores[isbn] = score

    top_isbns = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:n]
    result = books_df[books_df["isbn"].isin(top_isbns)][["isbn","title","author","year","publisher"]].copy()
    result = result.drop_duplicates(subset=["isbn"])
    result["hybrid_score"]  = result["isbn"].map(hybrid_scores)
    result["svd_score"]     = result["isbn"].map(svd_scores)
    result["content_score"] = result["isbn"].map(content_scores)
    result["graph_score"]   = result["isbn"].map(graph_scores)
    result = result.fillna(0).sort_values("hybrid_score", ascending=False).reset_index(drop=True)
    return result, liked_isbn


# ── Render Book Card HTML ───────────────────────────────────────
def render_book_card(row, rank, max_score):
    title       = html.escape(str(row["title"]))
    author      = html.escape(str(row["author"]))
    year        = html.escape(str(row.get("year", "N/A")))
    publisher   = html.escape(str(row.get("publisher", "Unknown")))
    isbn        = str(row["isbn"]).strip()

    # Dynamic Open Library Cover Art with generic fallback
    cover_url   = f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg"
    fallback_img = "https://images.unsplash.com/photo-1544947950-fa07a98d237f?auto=format&fit=crop&q=80&w=350"

    raw_score   = float(row["hybrid_score"])
    match_pct   = int((raw_score / max_score) * 100) if max_score > 0 else 50
    match_pct   = min(99, max(15, match_pct))

    svd_pct     = int(min(1.0, max(0.0, float(row["svd_score"]))) * 100)
    cnt_pct     = int(min(1.0, max(0.0, float(row["content_score"]))) * 100)
    grp_pct     = int(min(1.0, max(0.0, float(row["graph_score"]))) * 100)

    # Color coded gradient for badge
    if match_pct >= 80:
        badge_grad = "linear-gradient(135deg, #10b981 0%, #059669 100%)"
    elif match_pct >= 55:
        badge_grad = "linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)"
    else:
        badge_grad = "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)"

    card_html = f"""
    <div class="book-card">
        <div class="cover-wrapper">
            <img class="book-cover" src="{cover_url}" alt="{title}" loading="lazy"
                 onerror="this.onerror=null; this.src='{fallback_img}';" />
            <div class="rank-badge">#{rank}</div>
            <div class="score-badge" style="background: {badge_grad};">{match_pct}% Match</div>
        </div>
        <div class="card-details">
            <h4 class="book-title" title="{title}">{title}</h4>
            <div class="book-author">by {author}</div>
            <div class="meta-row">
                <span class="pill-meta">📅 {year}</span>
                <span class="pill-meta" title="{publisher}">🏢 {publisher[:18]}</span>
                <span class="pill-meta">🔑 {isbn}</span>
            </div>
            <div class="breakdown-box">
                <div class="meter-label-row">
                    <span>Collaborative (SVD)</span>
                    <span>{row['svd_score']:.2f}</span>
                </div>
                <div class="meter-track"><div class="fill-svd" style="width: {svd_pct}%;"></div></div>
                <div class="meter-label-row">
                    <span>Content (TF-IDF)</span>
                    <span>{row['content_score']:.2f}</span>
                </div>
                <div class="meter-track"><div class="fill-content" style="width: {cnt_pct}%;"></div></div>
                <div class="meter-label-row">
                    <span>Graph (A* Search)</span>
                    <span>{row['graph_score']:.2f}</span>
                </div>
                <div class="meter-track"><div class="fill-graph" style="width: {grp_pct}%;"></div></div>
            </div>
        </div>
    </div>
    """
    return card_html


# ── Load Data ───────────────────────────────────────────────────
df, books_df, svd_data, tfidf_matrix, title_to_idx, G = load_models()

# Curated iconic books for quick 1-click exploration
FEATURED_BOOKS = [
    "The Da Vinci Code",
    "To Kill a Mockingbird",
    "Pride and Prejudice",
    "The Hobbit (Leatherette Collector's Edition)",
    "1984"
]
available_featured = [b for b in FEATURED_BOOKS if any(b.lower() in t for t in title_to_idx)]


# ── Hero Banner ─────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <div class="hero-badge">✨ Hybrid AI Engine v2.0</div>
    <h1 class="hero-title">NextRead — AI Book Recommendation Studio</h1>
    <div class="hero-subtitle">
        Discover books tailored to your unique taste. Combining <b>Matrix Factorization (SVD)</b> for collaborative taste, 
        <b>TF-IDF Vector Space</b> for thematic story similarity, and <b>A* Graph Search</b> for serendipitous exploration.
    </div>
    <div class="hero-stats-row">
        <div class="stat-chip">📚 <b>271,000+</b> Books Catalog</div>
        <div class="stat-chip">👥 <b>Collaborative</b> User Profiling</div>
        <div class="stat-chip">🕸️ <b>Co-Rating</b> Similarity Graph</div>
        <div class="stat-chip">🖼️ <b>Live Cover Art</b> Integration</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar Controls ────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Recommendation Studio")

# User ID selection with random generator
all_users = sorted(df["user_id"].unique().tolist())
if "selected_user" not in st.session_state:
    st.session_state.selected_user = all_users[0]

col_u1, col_u2 = st.sidebar.columns([3, 1])
with col_u1:
    user_index = all_users.index(st.session_state.selected_user) if st.session_state.selected_user in all_users else 0
    selected_user = st.selectbox("Select User Profile", all_users[:500], index=user_index, key="user_select_box")
    st.session_state.selected_user = selected_user

with col_u2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button("🎲", help="Pick a random active user"):
        st.session_state.selected_user = random.choice(all_users[:500])
        st.rerun()

st.sidebar.divider()

# Book Selection
st.sidebar.markdown("### 📖 Seed Book Preference")

# Quick chips for featured titles
st.sidebar.caption("Quick picks:")
chip_cols = st.sidebar.columns(len(available_featured))
for idx, title_chip in enumerate(available_featured):
    # Shorten chip label
    short_label = title_chip.split("(")[0].strip()
    if short_label.startswith("The "):
        short_label = short_label[4:]
    if chip_cols[idx].button(short_label[:9] + "…", key=f"chip_{idx}", help=f"Set seed to '{title_chip}'"):
        st.session_state.seed_book = title_chip
        st.rerun()

all_titles = sorted(books_df["title"].dropna().unique().tolist())
if "seed_book" not in st.session_state:
    st.session_state.seed_book = available_featured[0] if available_featured else all_titles[0]

book_index = all_titles.index(st.session_state.seed_book) if st.session_state.seed_book in all_titles else 0
liked_book = st.sidebar.selectbox("Choose or search a book you like", all_titles, index=book_index)
st.session_state.seed_book = liked_book

st.sidebar.divider()

# Preset Strategies & Weight Sliders
st.sidebar.markdown("### 🎛️ Hybrid Strategy Presets")
preset_choice = st.sidebar.radio(
    "Choose weighting profile:",
    ["Balanced Hybrid", "Collaborative Focus", "Content & Theme Focus", "Serendipity & Novelty"],
    index=0
)

if preset_choice == "Balanced Hybrid":
    def_alpha, def_beta, def_gamma = 0.40, 0.40, 0.20
elif preset_choice == "Collaborative Focus":
    def_alpha, def_beta, def_gamma = 0.70, 0.20, 0.10
elif preset_choice == "Content & Theme Focus":
    def_alpha, def_beta, def_gamma = 0.15, 0.70, 0.15
else:
    def_alpha, def_beta, def_gamma = 0.20, 0.20, 0.60

with st.sidebar.expander("Fine-Tune Weight Sliders", expanded=False):
    alpha = st.slider("Collaborative Filtering (SVD)", 0.0, 1.0, def_alpha, 0.05)
    beta  = st.slider("Content-Based (TF-IDF)", 0.0, 1.0, def_beta, 0.05)
    gamma = st.slider("Graph Search (A*)", 0.0, 1.0, def_gamma, 0.05)

n_recs = st.sidebar.slider("Number of recommendations", min_value=3, max_value=24, value=9, step=3)

# Action button
st.sidebar.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
generate_clicked = st.sidebar.button("🚀 Generate Recommendations", type="primary", use_container_width=True)

# Keep recommendations in session state so switching tabs or UI doesn't reset them
if "current_recs" not in st.session_state:
    st.session_state.current_recs = None
    st.session_state.last_query = None


# ── Execution Logic ─────────────────────────────────────────────
if generate_clicked:
    with st.spinner("🤖 Running SVD, TF-IDF, and A* Graph traversal engines..."):
        recs, seed_isbn = hybrid_recommend(
            st.session_state.selected_user,
            liked_book,
            df,
            books_df,
            svd_data,
            tfidf_matrix,
            title_to_idx,
            G,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            n=n_recs
        )
        st.session_state.current_recs = recs
        st.session_state.last_query = {
            "user_id": st.session_state.selected_user,
            "book": liked_book,
            "isbn": seed_isbn,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "n": n_recs
        }


# ── Main Content Presentation ───────────────────────────────────
if st.session_state.current_recs is not None:
    recs  = st.session_state.current_recs
    query = st.session_state.last_query

    # Query Context Banner
    st.markdown(f"""
    <div class="query-banner">
        <h4 style="margin: 0 0 4px 0; color: #4338ca;">🎯 Recommendations for <b>User {query['user_id']}</b></h4>
        <div style="font-size: 0.92rem; color: #475569;">
            Based on seed title: <b>"{query['book']}"</b> &nbsp;|&nbsp; 
            Weights: <b>SVD {int(query['alpha']*100)}%</b> • 
            <b>TF-IDF {int(query['beta']*100)}%</b> • 
            <b>Graph {int(query['gamma']*100)}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_shelf, tab_comparison, tab_engines, tab_table = st.tabs([
        "📚 Visual Bookshelf",
        "📊 Score Analytics",
        "⚙️ Individual Engine Scores",
        "📋 Data & Export"
    ])

    # ── TAB 1: Visual Bookshelf (Grid with Dynamic Covers) ─────
    with tab_shelf:
        max_score = recs["hybrid_score"].max() if not recs.empty else 1.0
        
        # 3 cards per row for wide layout
        cols_per_row = 3
        for row_start in range(0, len(recs), cols_per_row):
            cols = st.columns(cols_per_row)
            for idx in range(cols_per_row):
                item_idx = row_start + idx
                if item_idx < len(recs):
                    row = recs.iloc[item_idx]
                    card_markup = render_book_card(row, item_idx + 1, max_score)
                    cols[idx].markdown(card_markup, unsafe_allow_html=True)

    # ── TAB 2: Score Analytics ──────────────────────────────────
    with tab_comparison:
        st.markdown("### 📊 Algorithmic Contribution Breakdown")
        st.caption("Compare how each engine contributed to the final hybrid score for each recommended title.")
        
        chart_data = recs[["title", "svd_score", "content_score", "graph_score", "hybrid_score"]].copy()
        # Truncate long titles for readable chart
        chart_data["short_title"] = chart_data["title"].apply(lambda t: t[:28] + "…" if len(t) > 28 else t)
        chart_display = chart_data.set_index("short_title")[["svd_score", "content_score", "graph_score", "hybrid_score"]]
        chart_display.columns = ["Collaborative (SVD)", "Content (TF-IDF)", "Graph Search (A*)", "Hybrid Match"]
        
        st.bar_chart(chart_display, height=380)

        col_m1, col_m2, col_m3 = st.columns(3)
        top_svd = recs.loc[recs["svd_score"].idxmax()]["title"] if not recs.empty else "N/A"
        top_content = recs.loc[recs["content_score"].idxmax()]["title"] if not recs.empty else "N/A"
        top_graph = recs.loc[recs["graph_score"].idxmax()]["title"] if not recs.empty else "N/A"

        col_m1.metric("Highest Collaborative Match", top_svd[:25] + "…", f"Score: {recs['svd_score'].max():.3f}")
        col_m2.metric("Highest Content Similarity", top_content[:25] + "…", f"Score: {recs['content_score'].max():.3f}")
        col_m3.metric("Highest Graph Connectivity", top_graph[:25] + "…", f"Score: {recs['graph_score'].max():.3f}")

    # ── TAB 3: Individual Engine Rankings ───────────────────────
    with tab_engines:
        st.markdown("### 🔬 Independent Model Rankings")
        sub_col1, sub_col2, sub_col3 = st.columns(3)

        with sub_col1:
            st.markdown("#### 👥 SVD Collaborative")
            st.caption("Learned from community user rating patterns")
            s_df = recs[["title", "author", "svd_score"]].sort_values("svd_score", ascending=False).reset_index(drop=True)
            st.dataframe(s_df, use_container_width=True)

        with sub_col2:
            st.markdown("#### 📖 TF-IDF Content")
            st.caption("Text similarity across plot & metadata")
            c_df = recs[["title", "author", "content_score"]].sort_values("content_score", ascending=False).reset_index(drop=True)
            st.dataframe(c_df, use_container_width=True)

        with sub_col3:
            st.markdown("#### 🕸️ A* Graph Search")
            st.caption("Traversal over book co-occurrence network")
            g_df = recs[["title", "author", "graph_score"]].sort_values("graph_score", ascending=False).reset_index(drop=True)
            st.dataframe(g_df, use_container_width=True)

    # ── TAB 4: Data Table & CSV Export ──────────────────────────
    with tab_table:
        st.markdown("### 📋 Complete Recommendation Records")
        export_df = recs[["isbn", "title", "author", "year", "publisher", "hybrid_score", "svd_score", "content_score", "graph_score"]].copy()
        export_df.columns = ["ISBN", "Title", "Author", "Year", "Publisher", "Hybrid Score", "SVD Score", "Content Score", "Graph Score"]
        st.dataframe(export_df, use_container_width=True)

        csv_data = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Recommendations as CSV",
            data=csv_data,
            file_name=f"recommendations_user_{query['user_id']}.csv",
            mime="text/csv",
            type="primary"
        )

else:
    # ── Welcome & How it Works State ────────────────────────────
    st.markdown("### 💡 Welcome to NextRead")
    st.info("👈 Select a user profile and a seed book in the sidebar, or pick one of the featured titles below to begin.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 👥 Collaborative Filtering (SVD)
        - **Algorithm:** Truncated Singular Value Decomposition
        - **Source:** Over 1 Million user ratings
        - **Strength:** Discovers hidden personal taste affinities across similar reading profiles.
        """)
    with col2:
        st.markdown("""
        #### 📖 Content Matching (TF-IDF)
        - **Algorithm:** Term Frequency - Inverse Document Frequency & Cosine Similarity
        - **Source:** Book titles, authors, and publisher metadata
        - **Strength:** Pinpoints books with identical thematic tone, narrative topics, and author styles.
        """)
    with col3:
        st.markdown("""
        #### 🕸️ Graph Traversal (A* Search)
        - **Algorithm:** Weighted NetworkX Graph Search
        - **Source:** Reader co-rating network
        - **Strength:** Unlocks serendipitous recommendations outside obvious clusters to avoid echo-chambers.
        """)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### 🌟 Featured Classics Ready to Explore")
    
    # Render featured sample cards
    feat_cols = st.columns(len(available_featured))
    for i, f_title in enumerate(available_featured):
        match_row = books_df[books_df["title"] == f_title]
        if not match_row.empty:
            f_isbn = match_row.iloc[0]["isbn"]
            f_author = match_row.iloc[0]["author"]
            f_year = match_row.iloc[0]["year"]
            with feat_cols[i]:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(140,140,160,0.2); border-radius: 12px; padding: 12px; text-align: center;">
                    <img src="https://covers.openlibrary.org/b/isbn/{f_isbn}-M.jpg" 
                         style="width: 100%; height: 180px; object-fit: cover; border-radius: 8px; margin-bottom: 8px;"
                         onerror="this.onerror=null; this.src='https://images.unsplash.com/photo-1544947950-fa07a98d237f?auto=format&fit=crop&q=80&w=250';" />
                    <div style="font-weight: 700; font-size: 0.85rem; height: 2.2rem; overflow: hidden;">{f_title[:32]}</div>
                    <div style="font-size: 0.75rem; color: #6366f1;">{f_author} ({f_year})</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Explore", key=f"feat_btn_{i}", use_container_width=True):
                    st.session_state.seed_book = f_title
                    st.rerun()
