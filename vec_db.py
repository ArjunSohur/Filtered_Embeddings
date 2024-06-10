import sqlite3
from newspaper import Article
from sentence_transformers import SentenceTransformer
import torch
import multiprocessing
from ast import literal_eval
from tqdm import tqdm
import os
from typing import List, Tuple, Optional
from datetime import datetime

# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# Database Functions                                                           #
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #

def create_db() -> None:
    """
    Creates a database table named 'embeddings' in a file named 'embeddings.db' to store scraped article information and embeddings (if applicable).

    The table has the following columns:
        * url (TEXT, PRIMARY KEY): Unique identifier for the article (the URL).
        * embedding (BLOB): Stores the article's embedding (if it's generated). This column can be null.
        * text (TEXT): Full text content of the article (if available). This column can be null.
        * source (TEXT): Source of the article (e.g., news website name).
        * authors (TEXT): Comma-separated list of the article's authors (if available). This column can be null.
        * title (TEXT): Title of the article.
        * publication_date (TEXT): Publication date of the article.

    This function ensures the table exists using `CREATE TABLE IF NOT EXISTS`, so it can be called repeatedly without creating duplicate tables.

    Returns:
        None
    """
    conn: sqlite3.Connection = sqlite3.connect('embeddings.db')
    c: sqlite3.Cursor = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings
                 (url TEXT PRIMARY KEY, 
                  embedding TEXT,
                  text TEXT,
                  source TEXT,
                  authors TEXT,
                  title TEXT,
                  publication_date TEXT)''')
    conn.commit()
    conn.close()

def store_in_db(url: str, embedding: torch.Tensor, text: str, source: str, authors: str, title: str, publication_date: Optional[str]) -> None:
    """
    Stores information about a scraped article and its embedding (if available) in the 'embeddings' table of a database named 'embeddings.db'.

    This function uses `INSERT OR REPLACE` to ensure that if an article with the same URL already exists, its information is updated with the provided data.

    Returns:
        None
    """
    conn: sqlite3.Connection = sqlite3.connect('embeddings.db')
    c: sqlite3.Cursor = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO embeddings 
                 (url, embedding, text, source, authors, title, publication_date) 
                 VALUES (?, ?, ?, ?, ?, ?, ?)''', 
              (url, embedding, text, source, ', '.join(authors), title, publication_date))
    conn.commit()
    conn.close()

# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# Processing Links                                                             #
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #

def process_links_chunk(links_chunk: List[Tuple[str, str]], embedder: SentenceTransformer, thread: int) -> None:
    """
    Processes a chunk of links from a larger list of scraped links. It iterates over each link, 
    calls the `process_link` function to handle individual link processing (likely involving 
    downloading content and generating an embedding), and keeps track of the processed links.

    Args:
        links_chunk: A list of tuples containing the feed name (str) and the article URL (str).
        embedder: A Sentence Transformer model used for generating embeddings (SentenceTransformer).
        thread: The ID of the current thread processing the link chunk (int).

    This function prints progress information every 100 processed links, indicating the number 
    of links completed and the thread ID.

    Returns:
        None
    """
    count: int = 0

    for link in links_chunk:
        process_link(link, embedder)
        count += 1

        if count % 50 == 0:
            print(f"Completed {count} links in thread {thread}")

def process_link(link: Tuple[str, str], embedder: SentenceTransformer) -> None:
    """
    Processes a single link from a list of scraped links. It performs the following steps:

    Args:
        link: A tuple containing the feed name (str) and the article URL (str).
        embedder: A Sentence Transformer model used for generating embeddings (SentenceTransformer).

    This function handles potential exceptions during download or parsing by printing an error message with the URL and the exception details.

    Returns:
        None
    """
    name: str = link[0]
    url: str = link[1]
    article: Article = Article(url)
    try:
        article.download()
        article.parse()
        text: str = article.text
        authors: List[str] = article.authors
        title: str = article.title
        publication: Optional[str] = article.publish_date.strftime("%Y-%m-%d") if article.publish_date else None

        embedding: str = repr(embedder.encode(text, convert_to_tensor=True))

        store_in_db(url, embedding, text, name, authors, title, publication) 
    except Exception as e:
        print(f"Failed to process {url}: {e}")

# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# Loading Embedder                                                             #
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #

def load_custom_model(model_name_or_path: str, cache_folder: str) -> SentenceTransformer:
    """
    Loads a SentenceTransformer model (pre-trained or custom).

    Args:
        model_name_or_path: Model name (pre-trained) or path (custom) (str).
        cache_folder: Directory to cache downloaded models (str).

    Downloads if missing, then loads the model.

    Returns:
        Loaded SentenceTransformer model (SentenceTransformer).
    """
    model_path: str = os.path.join(cache_folder, model_name_or_path)

    if not os.path.exists(model_path):
        print(f"Model '{model_name_or_path}' not found at '{model_path}'. Downloading...\n")
        
        os.makedirs(model_path, exist_ok=True)

        model: SentenceTransformer = SentenceTransformer(model_name_or_path, cache_folder=cache_folder, trust_remote_code=True, device="cpu")
        model.save(model_path)

        print("Downloading Complete, processing links ...\n")
    else:
        print(f"Model '{model_name_or_path}' found at '{model_path}'. Loading...")
        model = SentenceTransformer(model_name_or_path, cache_folder=cache_folder, trust_remote_code=True, device="cpu")
        print("Loading Complete, processing links ...\n")
    return model

# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# Main Process                                                                        #
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #

def store_vectors(links: List[Tuple[str, str]], embedding_model: str) -> None:
    """
    Stores embeddings for scraped links using multiprocessing.

    Args:
        links: List of tuples containing feed name (str) and URL (str).
        embedding_model: Name of pre-trained model or path to custom model (str).

    Creates a database, loads the embedding model, and processes links in chunks using multiple processes.
    """
    create_db()

    start_time: datetime = datetime.now()

    embedder: SentenceTransformer = load_custom_model(embedding_model, "local_models")
    if embedder is None:
        return

    num_multiprocessers: int = 4
    chunk_size: int = len(links) // num_multiprocessers + 1

    processes: List[multiprocessing.Process] = []
    for i in range(num_multiprocessers):
        start: int = i * chunk_size
        end: int = min((i + 1) * chunk_size, len(links))
        chunk: List[Tuple[str, str]] = links[start:end]
        p: multiprocessing.Process = multiprocessing.Process(target=process_links_chunk, args=(chunk, embedder, i+1))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    
    end: datetime = datetime.now()


    print("Time taken to process", len(links), "Articles:", str(end-start_time))

# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# Main                                                                         #
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    links: List[Tuple[str, str]] = []

    with open("links.txt", "r") as f:
        links = f.readlines()
        for i in range(len(links)):
            links[i] = tuple(links[i].strip().split(", "))

    embedding_model: str = "Alibaba-NLP/gte-large-en-v1.5"

    store_vectors(links, embedding_model)
