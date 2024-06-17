

def get_news_report_prompt(data, question, length="short"):
    text = list(data["text"])
    titles = list(data["title"])
    authors = list(data["authors"])
    urls = list(data["url"])
    
    context = ""
    for i in range(len(text)):
        context += f"Title: {titles[i]}\n\nText: {text[i]}\n------------------\n"

    prompt = f"""Here is what you will report on: {question}
    
    and here is the context: 
    {context}

    Your report should be {length}."""

    system_prompt = """You are an expert on all things news and current events.
    
    As such, people come and ask you questions about what is happening in the
    world.  You will get either a question or the name of a news event, and you
    must provide a comprehensive report on the topic using ONLY the provided
    context.  They will also give you a notion of how long they want the report
    to be.

    Short ~ your response should be bullet points
    Medium ~ your response should be about 5-6 sentences and just hit on main points.
    Long ~ as many as you need to felsh out the topic.
    
    You will have context to help you answer the question, but you must use your
    own words since you want to respect the original author's work.
    
    If you don't have sufficient or relevant context to answer the request, you can simply
    say that you don't have enough information to provide a response.
    
    Your response should just fullfill their request and nothing more."""

    sources = "SOURCES:\n"

    for i in range(len(urls)):
        author_str = ""

        if len(authors[i]) == 0:
            author_str = "Unknown"
        elif isinstance(authors[i], str):
            author_str = authors[i]
        else:
            author_str = ", ".join(authors[i])

        sources += f"  - \"{titles[i]}\" by {author_str}\n    {urls[i]}\n"
    
    return prompt, system_prompt, sources
    

