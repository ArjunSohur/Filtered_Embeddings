

def get_news_report_prompt(data, question, length="short"):
    text = list(data["text"])
    titles = list(data["title"])
    authors = list(data["authors"])
    urls = list(data["url"])

    length_dict = {"short": "three bullet points", "medium": "6-7 sentences", "long": "as many as you need to flesh out the topic"}
    reponse_length = length_dict[length]
    
    context = ""
    for i in range(len(text)):
        context += f"Title: {titles[i]}\n\nText: {text[i]}\n------------------\n"

    prompt = f"""Here is what you will report on: {question}
    
    and here is the context: 
    {context}

    Your report should be {reponse_length}, unless they ask a direct question,
    in which case, just answer the question."""

    system_prompt = """You are an expert on all things news and current events.
    
    As such, people come and ask you questions about what is happening in the
    world.  You will get either a question or the name of a news event, and you
    must provide a comprehensive report on the topic using ONLY the provided
    context.  They will also give you a notion of how long they want the report
    to be.
    
    You will have context to help you answer the question, but you must use your
    own words since you want to respect the original author's work.  You should
    NOT organize your report based by article, but rather, make it flow in a 
    cohesive way.
    
    If you don't have sufficient or relevant context to answer the request, you can simply
    say that you don't have enough information to provide a response.
    
    Your response should just fullfill the requirements and nothing more.  Do
    not mention anything about the requiremnts in your response and do NOT have 
    a lead in, e.g. 'here's my response:'.
    
    You are very determined to write a good and comprehensive report."""

    sources = "SOURCES:\n"

    for i in range(len(urls)):
        author_str = ""

        if len(authors[i]) == 0:
            author_str = "Unknown"
        elif isinstance(authors[i], str):
            author_str = authors[i]
        else:
            author_str = ", ".join(authors[i])

        sources += f"  • \"{titles[i]}\" by {author_str}\n    {urls[i]}\n"
    
    return prompt, system_prompt, sources
    

