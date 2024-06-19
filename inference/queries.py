import torch
import pandas as pd
import sqlite3
from ast import literal_eval


# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# Get similar                                                                  #                       
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
def get_similar(text, data, embedder, top_n=5, threshold=0.5, sql_path = "embeddings.db", blacklist: list[str] = [], whitelist: list[str] = [], wl_boost: dict = []) -> pd.DataFrame:
    text_vetor = torch.Tensor(embedder.encode(text))

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
    

