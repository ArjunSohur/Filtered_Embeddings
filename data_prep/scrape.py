"""
The goal of this file is to scrape all the RSS feeds and get all the links from
them in the least amount of time possible whilst hopefully using a reasonable
amount of computation.
"""
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# Imports                                                                      #                       
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
import feedparser
import multiprocessing
from datetime import datetime, timedelta
import random

# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# Get Feeds                                                                    #                       
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
def get_feeds() -> list[tuple[str, str]]:
  """
  Reads RSS feed information (name and URL) from a text file and returns a list of tuples containing this information.

  Returns:
      list[tuple[str, str]]: A list of tuples where each tuple contains the feed name and URL.
  """
  with open('data_prep/feeds.txt', 'r') as f:
    feeds: list[str] = f.readlines()

  feed_info: list[tuple[str, str]] = []

  for feed in feeds:
    feed = feed.strip()
    duo = feed.split(", ")
    feed_info.append((duo[0], duo[1]))

  return feed_info


# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# RSS info-getting                                                             # 
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
def parse_section(
    start: int, end: int, feeds: list[tuple[str, str]], link_list: list[tuple[str, str]]) -> None:
  """
  A helper function used for parallel parsing. It takes a starting index, ending index, list of feeds, and a link list as arguments.
  It iterates over the assigned chunk of feeds, parses entries for entries published in the last 48 hours, and appends them to the link list.

  Args:
      start: The starting index of the feed chunk to process (int).
      end: The ending index (exclusive) of the feed chunk to process (int).
      feeds: The list of RSS feeds to parse (list[tuple[str, str]]).
      link_list: list of links...

  Returns:
      None
  """
  links: list[tuple[str, str]] = []
  current_time: datetime = datetime.now() 
  time_threshold: datetime = current_time - timedelta(hours=48)  

  for i in range(start, end):
    feed = feeds[i]
    name: str = feed[0]  # Name of the RSS feed
    url: str = feed[1]  # URL of the RSS feed
    d = feedparser.parse(url)  

    for entry in d.entries:
      published_time: datetime | None = None  

      try:
        if "published_parsed" in entry and entry.published_parsed:
          published_time = datetime(*entry.published_parsed[:6])
        elif "updated_parsed" in entry and entry.updated_parsed:
          published_time = datetime(*entry.updated_parsed[:6])
      except Exception as e:
        print(f"Error parsing date for {entry.link}: {e}")

      if published_time and published_time > time_threshold:
        links.append((name, entry.link)) 

  link_list.extend(links)  


def parse_rss(feeds: list[tuple[str, str]]) -> list[tuple[str, str]]:
  """
  This function takes a list of feeds as input. It creates a specified number of worker processes using multiprocessing.
  Each process calls the `parse_section` function to process a chunk of feeds concurrently. Finally, it combines the results from all processes and returns a list of scraped links.

  Args:
      feeds: The list of RSS feeds to parse (list[tuple[str, str]]).

  Returns:
      list[tuple[str, str]]: A list of tuples containing the feed name and scraped link.
  """
  # I've found that 4 is the fastest for my mac - yours might be different
  num_multiprocessers: int = 4
  chunk_size: int = len(feeds) // num_multiprocessers + 1

  with multiprocessing.Manager() as manager:
    link_list: multiprocessing.Manager.list = manager.list()
    processes: list[multiprocessing.Process] = []

    for i in range(num_multiprocessers):
      start: int = i * chunk_size
      end: int = min((i + 1) * chunk_size, len(feeds))

      p: multiprocessing.Process = multiprocessing.Process(target=parse_section, args=(start, end, feeds, link_list), name=f"Process {i+1}")
      processes.append(p)
      p.start()

    for p in processes:
      p.join()

    links: list[tuple[str, str]] = list(link_list)

  return links

# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# Scrape                                                                       #
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
def scrape(num_feeds = None) -> list[tuple[str, str]]:
    """
    The main function of the script. It calls `get_feeds` to retrieve RSS feed information, then calls `parse_rss` to scrape links from those feeds in parallel.
    Finally, it prints the time taken to scrape all the links and returns the scraped links.

    Returns:
        list[tuple[str, str]]: A list of tuples containing the feed name and scraped link.
    """
    start_time: datetime = datetime.now()
    print("Starting Scraping of RSS feeds")

    if num_feeds:
      feeds: list[tuple[str, str]] = get_feeds()
      random.shuffle(feeds)
      feeds = feeds[:num_feeds]
    else:
      feeds: list[tuple[str, str]] = get_feeds()

    links: list[tuple[str, str]] = parse_rss(feeds)  

    end: datetime = datetime.now() 

    print("Time taken to get", len(links), "RSS links:", str(end-start_time))

    return links



# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
# Main                                                                         #
# ---------------------------------------------------------------------------- #
#                                                                              #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    scrape()

