# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# Imports                                                                      #
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
import pandas as pd
from pandas.core.frame import DataFrame

from scrape import scrape
from vec_db import store_vectors

from queries import sql3_as_pd


# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# Scraping                                                                     #
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
def f_scrape(bool) -> list[tuple[str, str]]:
    if bool:
        links = scrape()

        links = list(set(links))

        with open("links.txt", "w") as f:
            for name, link in links:
                f.write(f"{name}, {link}\n")
    else:
        with open("links.txt", "r") as f:
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
def f_store(bool, links):
    if bool:
        embedding_model = "Alibaba-NLP/gte-large-en-v1.5"

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
    q_scrape = False
    q_store = True

    links: list[tuple[str, str]] = f_scrape(q_scrape)

    links = links[:10]

    f_store(q_store, links)

    # data: DataFrame = sql3_as_pd("embeddings.db")
    