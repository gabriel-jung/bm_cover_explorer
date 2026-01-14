import streamlit as st
import pandas as pd
import numpy as np
import torch
import open_clip
import random

# --- CONFIG & SETUP ---
st.set_page_config(layout="wide", page_title="BM Cover Explorer")


@st.cache_resource
def load_torch_assets():
    """Load model, tokenizer, and preprocess"""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Using SigLIP
    model_name = "ViT-B-16-SigLIP"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="webli", device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, tokenizer, device


@st.cache_data
def load_cover_embeddings(path):
    """Load the cover embeddings from the pickle file"""
    df = pd.read_pickle(path)
    matrix = np.vstack(df["embedding"].values).astype("float32")
    return df, matrix


@st.cache_data
def get_text_embedding(text, _model, _tokenizer, _device):
    """Encode text to the same embedding space as the covers"""
    tokens = _tokenizer([text]).to(_device)
    with torch.no_grad():
        feat = _model.encode_text(tokens)
        feat /= feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()


# --- SESSION STATE INITIALIZATION ---
if "search_active" not in st.session_state:
    st.session_state.search_active = False
if "random_idx" not in st.session_state:
    st.session_state.random_idx = None


def handle_random_click():
    st.session_state.search_active = True
    st.session_state.random_idx = random.randint(0, len(df) - 1)
    st.session_state.search_input = ""


# --- APP LOGIC ---
model, tokenizer, device = load_torch_assets()
df, embeddings_matrix = load_cover_embeddings("data/bm_covers_with_embeddings.pkl")

# UI Header
st.title("Black Metal Cover Explorer")
st.markdown("Search 3,000+ albums by describing elements of the cover artwork.")

# Search Controls
# --- SEARCH CONTROLS ---
col_a, col_mid, col_b = st.columns([8, 1, 2], vertical_alignment="bottom")

with col_a:
    query = st.text_input(
        "Visual description",
        placeholder="e.g., 'forest in winter'...",
        key="search_input",
    )

with col_mid:
    # 2026 Streamlit 1.52+ native text centering
    st.markdown(
        "**OR**",
        text_alignment="center",
    )

with col_b:
    # Ensure button has no top-margin weirdness
    st.button("🎲 Random Album", width="stretch", on_click=handle_random_click)

# Logic to determine if a search is active
if query:
    st.session_state.search_active = True
    st.session_state.random_idx = None

if not st.session_state.search_active:
    st.info("Enter a description or click 'Random' to get started!")
    st.stop()

# Determine Query Vector
if query:
    q_vec = get_text_embedding(query, model, tokenizer, device)
    title = f"Results for: '{query}'"
else:
    st.session_state.random_idx = random.randint(0, len(df) - 1)
    idx = st.session_state.random_idx
    q_vec = df.iloc[idx]["embedding"].reshape(1, -1)
    title = f"Similar to: {df.iloc[idx]['band']} - {df.iloc[idx]['album']}"

# --- CALC SIMILARITY ---
# Dot product of [1, 768] @ [768, N] -> [N] scores
scores = (q_vec @ embeddings_matrix.T).flatten()
sorted_indices = scores.argsort()[::-1]

# --- SEARCH SETTINGS ---
st.divider()
col_title, col_settings = st.columns([9, 2])

with col_title:
    st.subheader(title)

with col_settings:
    top_k = st.slider("Results shown", min_value=1, max_value=10, value=5)

with st.expander("🔧 Advanced Search Settings", expanded=False):
    set_a, set_b, set_c = st.columns(3)
    with set_a:
        show_scores = st.checkbox(
            "Show similarity scores",
            value=False,
            help="Display similarity scores between the query and each cover. Useful for comparison but absolute values are not meaningful.",
        )
    with set_b:
        unique_bands = st.checkbox(
            "Limit to one album per band",
            value=True,
            help="Ensure that only one album from each band appears in the results.",
        )
    with set_c:
        top_k_pool = st.slider(
            "Result pool size",
            min_value=top_k,
            max_value=100,
            value=top_k,
            help="Expands the search pool before picking results at random. Increase this to see a wider variety of albums instead of just the absolute top matches.",
        )
        if top_k_pool > top_k:
            st.caption(
                f"Picking {top_k} random albums from the top {top_k_pool} matches."
            )
        else:
            st.caption("Showing the most similar matches.")

# --- FILTERING ---

results_pool = []
seen_bands_pool = set()

exclude_idx = st.session_state.random_idx if not query else None

for idx in sorted_indices:
    if len(results_pool) >= top_k_pool:
        break

    if idx == exclude_idx:
        continue

    artist = df.iloc[idx]["band"]
    if unique_bands and artist in seen_bands_pool:
        continue

    results_pool.append(idx)
    seen_bands_pool.add(artist)

if top_k_pool > top_k:
    random.shuffle(results_pool)
    results_to_show = results_pool[:top_k]
    results_to_show = sorted(results_to_show, key=lambda x: scores[x], reverse=True)
else:
    results_to_show = results_pool[:top_k]

worst_indices = sorted_indices[::-1]


# --- REFERENCE ALBUM (Only for Random/Album-based search) ---
if not query and st.session_state.random_idx is not None:
    ref_idx = st.session_state.random_idx
    ref_row = df.iloc[ref_idx]

    with st.container(border=True):
        col_img, col_txt = st.columns([1, 3], vertical_alignment="center")
        with col_img:
            st.image(ref_row["cover_url"], width=200)
        with col_txt:
            st.write("### Currently Exploring:")
            st.markdown(
                f"**{ref_row['album']}** by **{ref_row['band']}** ({ref_row['year']})"
            )
            st.caption("Showing the most visually similar covers.")

# --- RESULTS GRID ---

cols = st.columns(5)
with st.container(horizontal=True, horizontal_alignment="left", gap="medium"):
    for i, idx in enumerate(results_to_show):
        row = df.iloc[idx]
        score = np.clip(float(scores[idx]), 0, 1)
        url = f"https://www.metal-archives.com/albums/_/_/{row['album_id']}"

        with cols[i % 5]:
            with st.container(border=True, height="stretch"):
                st.image(row["cover_url"], width="stretch")
                st.markdown(f"**[{row['album']}]({url})** ({row['year']})")
                st.markdown(f"{row['band']}")
                if show_scores:
                    st.progress(score, text=f"Score: {score:.3f}")


# --- DISTANT RESULTS SECTION ---
st.divider()

show_distant = st.checkbox(("🌑 Show  (Least Similar Covers)"), value=False)

if show_distant:
    results_to_avoid = []
    seen_bands_worst = set()
    for idx in worst_indices:
        if len(results_to_avoid) >= top_k:
            break
        artist = df.iloc[idx]["band"]
        if unique_bands and artist in seen_bands_worst:
            continue
        results_to_avoid.append(idx)
        seen_bands_worst.add(artist)

    st.info("These covers are statistically the **least similar** to your description.")
    cols = st.columns(5)
    with st.container(horizontal=True, horizontal_alignment="left", gap="medium"):
        for i, idx in enumerate(results_to_avoid):
            row = df.iloc[idx]
            score = np.clip(float(scores[idx]), 0, 1)
            url = f"https://www.metal-archives.com/albums/_/_/{row['album_id']}"
            with cols[i % 5]:
                with st.container(border=True, height="stretch"):
                    st.image(row["cover_url"], width="stretch")
                    st.markdown(f"**[{row['album']}]({url})**")
                    st.caption(f"{row['band']}")
                    if show_scores:
                        st.progress(score, text=f"Score: {score:.3f}")
