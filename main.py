# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# Imports                                                                      #
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
from scrape import scrape
from vec_db import store_vectors

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

    if q_scrape:
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

    if q_scrape:
        print("Scraping Complete\n")
    else:
        print("Link Collection Complete\n")
    
    if q_store:
        embedding_model = "Alibaba-NLP/gte-large-en-v1.5"

        store_vectors(links, embedding_model)

        print("Vectors Stored\n")
    else:
        print("Using existing data\n")
    
    

