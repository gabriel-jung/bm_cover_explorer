# BM Cover Explorer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bm-cover-explorer.streamlit.app)
[![Model: SigLIP](https://img.shields.io/badge/Model-SigLIP--B16-black?style=flat-square)](https://huggingface.co/timm/ViT-B-16-SigLIP)

**BM Cover Explorer** is a semantic search engine for Black Metal album art. Powered by SigLIP (Vision Transformer), it enables natural language discovery across 3,000+ covers by matching text descriptions directly to visual concepts.


## Features

- **Natural Language Search**: Retrieve covers by describing visual elements (e.g., *"Dark fortress in blue night"*).
- **Image Search**: Upload an image to find visually similar covers in the archive.
- **Similarity Discovery**: Explore the archive by finding covers visually related to a "seed" album, or view the least similar matches.
- **Click-to-Explore**: Use any result as a new seed to navigate the archive by visual similarity.
- **Year Range Filter**: Narrow results to a specific release period.
- **Shareable URLs**: Text searches are encoded in the URL (`?q=...`) for easy sharing.
- **CSV Export**: Download current results as a CSV file.


## Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Vision Model**: [ViT-B-16-SigLIP](https://huggingface.co/timm/ViT-B-16-SigLIP) (Pretrained on WebLI)
- **Data Source**: Curated selection of ~3,300 Black Metal albums from [MusicBrainz](https://musicbrainz.org/), [Encyclopaedia Metallum](https://www.metal-archives.com/) and [Cover Art Archive](https://coverartarchive.org/)
- **Python Libraries**:
  - `open_clip`: For interfacing with the SigLIP model and generating embeddings.
  - `torch`: Backend tensor framework for model inference.
  - `pandas` & `numpy`: For metadata management and vector similarity calculations.
  - `pillow`: For image processing and rendering.


## Local Setup

1. **Clone the repository**
    ```bash
    git clone https://github.com/gabriel-jung/bm-cover-explorer.git
    cd bm-cover-explorer
    ```

2. **Install Dependencies**, choose a method:

    **Option A**: using uv
    ```bash
    uv sync
    ```
    **Option B**: using pip
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**, launch the Streamlit server based on your installation method:

    **If using uv:**
    ```bash
    uv run streamlit run app.py
    ```
    **If using pip:**
    ```bash
    streamlit run app.py
    ```

## License

- **Code**: This project's source code is licensed under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, share, and modify it.
- **Artwork**: This project is for **educational purposes** only. All album artwork remains the property of the respective artists and record labels.
