import torch
import pandas as pd
import sqlite3
from ast import literal_eval






def sql3_as_pd(path: str) -> pd.DataFrame:
    sql = sqlite3.connect(path)

    temp = pd.read_sql_query("SELECT * FROM embeddings", sql)

    print(f"\n\nLOOK HERE: {temp['embedding'][0]}\n\n")

    temp["embedding"] = temp['embedding'].map(lambda x: literal_eval(x))

    print(temp["embedding"][0])

    return temp
    

