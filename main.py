# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# Imports                                                                      #
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
import pandas as pd
from pandas.core.frame import DataFrame

from data_prep.scrape import scrape
from data_prep.vec_db import store_vectors

from inference.queries import sql3_as_pd, get_similar

from sentence_transformers import SentenceTransformer
import os

# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# Embedder                                                                     #
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
def load_custom_sentence_transformer(model_name_or_path: str = "Alibaba-NLP_gte-large-en-v1.5") -> SentenceTransformer:
    """
    Loads a SentenceTransformer model (pre-trained or custom).

    Args:
        model_name_or_path: Model name (pre-trained) or path (custom) (str).

    Downloads if missing, then loads the model.

    Returns:
        Loaded SentenceTransformer model (SentenceTransformer).
    """
    # Construct the path to the torch cache directory in the user's home directory
    cache_folder = os.path.join(os.path.expanduser("~"), ".cache", "torch", "sentence_transformers")
    model_path = os.path.join(cache_folder, model_name_or_path)

    if not os.path.exists(model_path):
        print(f"Model '{model_name_or_path}' not found at '{model_path}'. Downloading...\n")
        
        os.makedirs(cache_folder, exist_ok=True)

        # I have device as cpu because I am running this on a mac - obviously, change this to gpu if you have a gpu
        model = SentenceTransformer(model_name_or_path, cache_folder=cache_folder, trust_remote_code=True, device="cpu")
        model.save(model_path)

        print("Downloading Complete, processing links ...\n")
    else:
        print(f"Model '{model_name_or_path}' found at '{model_path}'. Loading...")
        model = SentenceTransformer(model_path, cache_folder=cache_folder, trust_remote_code=True, device="cpu")
        print("Loading Complete, processing links ...\n")
    return model


# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# Scraping                                                                     #
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
def f_scrape(bool, num_feeds = None) -> list[tuple[str, str]]:
    if bool:
        links = scrape(num_feeds)

        links = list(set(links))

        with open("data_prep/links.txt", "w") as f:
            for name, link in links:
                f.write(f"{name}, {link}\n")
    else:
        with open("data_prep/links.txt", "r") as f:
            links = f.readlines()

            for i in range(len(links)):
                links[i] = tuple(links[i].strip().split(", "))

    if bool:
        print("Scraping Complete\n")
    else:
        print("Link Collection Complete\n")

    return links

# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# Storing                                                                      #
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
def f_store(bool, links, embedding_model):
    if bool:
        store_vectors(links, embedding_model)

        print("Vectors Stored\n")
    else:
        print("Using existing data\n")

# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# Main                                                                         #
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    q_scrape = True
    q_store = True

    links: list[tuple[str, str]] = f_scrape(q_scrape, num_feeds=5)

    embedder = load_custom_sentence_transformer()
    f_store(q_store, links, embedder)

    data: DataFrame = sql3_as_pd("embeddings.db")

    txt = "Russia in Ukraine"
    similar = get_similar(txt, data, embedder, top_n=5, threshold=0.5)

    print(similar)