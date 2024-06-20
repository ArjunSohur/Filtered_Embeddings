import torch
import pandas as pd
import sqlite3
from ast import literal_eval
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, Pipeline
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# Bias                                                                         #
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
def get_bias_decector():
    base_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--newsmediabias--UnBIAS-classification-bert")
    snapshots_path = os.path.join(base_path, "snapshots")

    # List all directories under snapshots
    if os.path.exists(snapshots_path):
        snapshot_dirs = sorted([d for d in os.listdir(snapshots_path) if os.path.isdir(os.path.join(snapshots_path, d))])
        if snapshot_dirs:
            latest_snapshot = snapshot_dirs[-1]  # Select the most recent snapshot
            model_path = os.path.join(snapshots_path, latest_snapshot)
        else:
            raise EnvironmentError(f"No snapshot directories found in '{snapshots_path}'.")
    else:
        raise EnvironmentError(f"Snapshots path '{snapshots_path}' does not exist.")

    # Check if the necessary files exist in the model_path
    necessary_files = ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]
    if all(os.path.exists(os.path.join(model_path, file)) for file in necessary_files):
        print(f"Bias model found at '{model_path}'. Loading...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        print(f"Necessary files not found in '{model_path}'. Downloading...\n")
        tokenizer = AutoTokenizer.from_pretrained("newsmediabias/UnBIAS-classification-bert")
        model = AutoModelForSequenceClassification.from_pretrained("newsmediabias/UnBIAS-classification-bert")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if device.type == "cuda" else -1)

    return classifier

def get_bias(text, classifier):
    tokenizer = classifier.tokenizer

    tokens = tokenizer.tokenize(text)
    if len(tokens) > 512:
        tokens = tokens[:509]

    truncated_text = tokenizer.convert_tokens_to_string(tokens)

    # Classify the truncated text
    result = classifier(truncated_text)

    if result[0]['label'] == 'Biased':
        return result[0]['score']
    else:
        return 1 - result[0]['score']

# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# Get similar                                                                  #                       
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
def get_similar(text, data, embedder, top_n=3, threshold=0.5, sql_path = "embeddings.db",\
                 blacklist: list[str] = [], whitelist: list[str] = [], wl_boost: dict = [], \
                    date_filter:dict = None) -> pd.DataFrame:
    text_vetor = torch.Tensor(embedder.encode(text))

    if date_filter:
        data = data[data["date"] >= date_filter["start"]] if "start" in date_filter else data
        data = data[data["date"] <= date_filter["end"]]  if "end" in date_filter else data

    sims = []
    for i in range(len(data)):
        data_vector = torch.Tensor(data.iloc[i]["embedding"]) # is it the best idea to have there as a tensor?

        sim = torch.nn.functional.cosine_similarity(text_vetor, data_vector, dim=0).item()
        if sim > threshold and data.iloc[i]["source"] not in blacklist:

            if data.iloc[i]["source"] in whitelist:
                sims.append((data.iloc[i]["embedding"], max(sim + wl_boost[data.iloc[i]["source"]]), 1))
            else:
                sims.append((data.iloc[i]["embedding"], sim))
            

    top_n_sims = sorted(sims, key=lambda x: x[1], reverse=True)[:top_n]

    con = sqlite3.connect(sql_path)
    cur = con.cursor()

    rows = []

    for tup in top_n_sims:
        vector, sim = tup

        cur.execute(f"SELECT * FROM embeddings WHERE embedding = '{vector}'")
        rows.append(cur.fetchone())
    
    posts_df = pd.DataFrame(rows, columns=data.columns)

    # get bias of each post and add it to the dataframe
    # classifier = get_bias_decector()
    # posts_df["bias"] = posts_df["text"].map(lambda x: get_bias(x, classifier))

    return posts_df


# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# As dataframe                                                                 #                       
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
def sql3_as_pd(path: str) -> pd.DataFrame:
    sql = sqlite3.connect(path)

    temp = pd.read_sql_query("SELECT * FROM embeddings", sql)

    temp["embedding"] = temp['embedding'].map(lambda x: literal_eval(x))

    return temp
    

