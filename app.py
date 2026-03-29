import streamlit as st
import pandas as pd
import numpy as np
import torch
import open_clip
import random
from PIL import Image

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
    return model, tokenizer, preprocess, device


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


def get_image_embedding(image, _model, _preprocess, _device):
    """Encode an uploaded image to the same embedding space as the covers"""
    tensor = _preprocess(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        feat = _model.encode_image(tensor)
        feat /= feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()


# --- SESSION STATE INITIALIZATION ---
if "search_active" not in st.session_state:
    st.session_state.search_active = False
if "random_idx" not in st.session_state:
    st.session_state.random_idx = None


def set_album_seed(idx):
    """Use a specific album's embedding as the query vector for similarity search."""
    st.session_state.search_active = True
    st.session_state.random_idx = idx
    st.session_state.search_input = ""
    st.query_params.clear()


def handle_random_click():
    set_album_seed(random.randint(0, len(df) - 1))


# --- APP LOGIC ---
model, tokenizer, preprocess, device = load_torch_assets()
df, embeddings_matrix = load_cover_embeddings("data/bm_covers_with_embeddings.pkl")

# --- PERMALINK SUPPORT ---
params = st.query_params
if "q" in params and not st.session_state.search_active:
    st.session_state.search_active = True
    st.session_state.search_input = params["q"]

# UI Header
st.title("Black Metal Cover Explorer")
st.markdown("Search 3,000+ albums by describing elements of the cover artwork.")

# --- SEARCH CONTROLS ---
col_a, col_mid, col_b = st.columns([8, 1, 2], vertical_alignment="bottom")

with col_a:
    query = st.text_input(
        "Visual description",
        placeholder="e.g., 'forest in winter'...",
        key="search_input",
    )

with col_mid:
    st.markdown(
        "**OR**",
        text_alignment="center",
    )

with col_b:
    st.button("🎲 Random Album", width="stretch", on_click=handle_random_click)

uploaded_image = st.file_uploader(
    "Or search by image",
    type=["png", "jpg", "jpeg", "webp"],
    key="image_input",
)

# Query priority: text > uploaded image > album seed (random/explore)
if query:
    st.session_state.search_active = True
    st.session_state.random_idx = None
    st.query_params["q"] = query
elif uploaded_image:
    st.session_state.search_active = True
    st.session_state.random_idx = None
    st.query_params.clear()

if not st.session_state.search_active:
    st.info("Enter a description, upload an image, or click 'Random' to get started!")
    st.stop()

# Determine Query Vector
if query:
    q_vec = get_text_embedding(query, model, tokenizer, device)
    title = f"Results for: '{query}'"
elif uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    q_vec = get_image_embedding(image, model, preprocess, device)
    title = "Results for uploaded image"
else:
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
    set_a, set_b, set_c, set_d = st.columns(4)
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
    with set_d:
        year_min = int(df["year"].min())
        year_max = int(df["year"].max())
        year_range = st.slider(
            "Year range",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
            help="Filter results to albums released within this year range.",
        )

# --- FILTERING ---

# Build a set of valid indices based on year filter
if year_range != (year_min, year_max):
    year_mask = (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
    valid_years = set(np.where(year_mask)[0])
else:
    valid_years = None

results_pool = []
seen_bands_pool = set()

# Exclude the seed album itself from results when exploring by album
exclude_idx = st.session_state.random_idx if not query else None

for idx in sorted_indices:
    if len(results_pool) >= top_k_pool:
        break

    if idx == exclude_idx:
        continue

    if valid_years is not None and idx not in valid_years:
        continue

    artist = df.iloc[idx]["band"]
    if unique_bands and artist in seen_bands_pool:
        continue

    results_pool.append(idx)
    seen_bands_pool.add(artist)

if top_k_pool > top_k:
    random.shuffle(results_pool)
results_to_show = results_pool[:top_k]

# --- RESULT COUNT BADGE ---
# Empirical threshold: scores below 0.3 are visually unrelated
SIMILARITY_THRESHOLD = 0.3
filtered_scores = (
    scores[np.array(list(valid_years))] if valid_years is not None else scores
)
n_above = int((filtered_scores > SIMILARITY_THRESHOLD).sum())
st.caption(f"{n_above:,} albums above similarity threshold ({SIMILARITY_THRESHOLD})")

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

n_cols = min(top_k, 5)
cols = st.columns(n_cols)
with st.container(horizontal=True, horizontal_alignment="left", gap="medium"):
    for i, idx in enumerate(results_to_show):
        row = df.iloc[idx]
        score = np.clip(float(scores[idx]), 0, 1)
        url = f"https://www.metal-archives.com/albums/_/_/{row['album_id']}"

        with cols[i % n_cols]:
            with st.container(border=True, height="stretch"):
                st.image(row["cover_url"], width="stretch")
                st.markdown(f"**[{row['album']}]({url})** ({row['year']})")
                st.markdown(f"{row['band']}")
                if show_scores:
                    st.progress(score, text=f"Score: {score:.3f}")
                st.button(
                    "Explore similar",
                    key=f"explore_{idx}",
                    on_click=set_album_seed,
                    args=(idx,),
                )

# --- DOWNLOAD RESULTS ---
download_data = []
for idx in results_to_show:
    row = df.iloc[idx]
    download_data.append(
        {
            "band": row["band"],
            "album": row["album"],
            "year": row["year"],
            "score": round(float(scores[idx]), 4),
            "url": f"https://www.metal-archives.com/albums/_/_/{row['album_id']}",
        }
    )
csv = pd.DataFrame(download_data).to_csv(index=False)
st.download_button("Download results as CSV", csv, "bm_results.csv", "text/csv")

# --- DISTANT RESULTS SECTION ---
st.divider()

show_distant = st.checkbox("🌑 Show Least Similar Covers", value=False)

if show_distant:
    worst_indices = sorted_indices[::-1]
    results_to_avoid = []
    seen_bands_worst = set()
    for idx in worst_indices:
        if len(results_to_avoid) >= top_k:
            break

        if valid_years is not None and idx not in valid_years:
            continue

        artist = df.iloc[idx]["band"]
        if unique_bands and artist in seen_bands_worst:
            continue
        results_to_avoid.append(idx)
        seen_bands_worst.add(artist)

    st.info("These covers are statistically the **least similar** to your description.")
    cols = st.columns(n_cols)
    with st.container(horizontal=True, horizontal_alignment="left", gap="medium"):
        for i, idx in enumerate(results_to_avoid):
            row = df.iloc[idx]
            score = np.clip(float(scores[idx]), 0, 1)
            url = f"https://www.metal-archives.com/albums/_/_/{row['album_id']}"
            with cols[i % n_cols]:
                with st.container(border=True, height="stretch"):
                    st.image(row["cover_url"], width="stretch")
                    st.markdown(f"**[{row['album']}]({url})**")
                    st.caption(f"{row['band']}")
                    if show_scores:
                        st.progress(score, text=f"Score: {score:.3f}")
