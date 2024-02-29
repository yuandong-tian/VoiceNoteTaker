from datetime import datetime
import requests
import json

import requests
import google.generativeai as genai
import re
import os

genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(m.name)

model = genai.GenerativeModel('gemini-pro') 

def search_twitter(keyword):
    url = "https://twitter-api45.p.rapidapi.com/search.php"

    querystring = {"query": keyword}

    headers = {
        "X-RapidAPI-Key": "65e7ed4373msh23ee0ce1c1cf7f8p158655jsn3ec0ee26f237",
        "X-RapidAPI-Host": "twitter-api45.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    return response.json()

def get_sentiment(query):
    data = search_twitter(query)

    # # save the data to a file
    # with open("data.json", "w") as f:
    #     json.dump(data, f)

    # load the data from the file
    # with open("data.json", "r") as f:
    #     data = json.load(f)

    input_data = []
    for entry in data["timeline"]:
        # convert the created_at in the format of [Wed Feb 28 22:22:43 +0000 2024] to a datetime object and compute how recent the post is
        # Note that +0000 means it is UTC time, and we need to convert it to the local time
        created_at = entry['created_at']
        # get utc time now 
        recency = datetime.now() - datetime.strptime(created_at, "%a %b %d %H:%M:%S +0000 %Y")
        # convert to local time
        if recency.days < 2:
            # remove the new line character
            text = entry["text"].replace("\n", " ")
            views = entry["views"]
            s = f"[{created_at}] {text} [{views}]" 
            input_data.append((s, recency))

    # sort the input data by recency
    input_data = input_data.sort(key=lambda x: x[1])
    # remove the recency
    input_data = [x[0] for x in input_data]
    # Only use the first 10 posts
    input_data = input_data[:10]

    prompt = '''
        Summarize the following posts regarding to the stock {stock}. Each row is a post with the following format:
        [timestamp] text [views]
        For each row, return 
        1. a sentiment in the scale of [-1, 1] (positive=1, neutral=0, negative=-1) regarding to {stock}; 
        2. a quality metric of the text in the scale of [0, 1] (good quality=1, bad quality=0); 
        3. a classification label of the text, chosen from ["advertisement", "news", "opinion", "question", "facts"].
        The overall output should be a json object with the following format:
        [
            dict(sentiment=1, quality=1, label="news"),   
            dict(sentiment=-0.2, quality=0.6, label="opinion")   
        ]
        one dict for each post.

        Here are the posts:
    '''

    while True:
        try:
            response = model.generate_content(prompt.format(stock=query) + "\n".join(input_data))
            # parse by json
            results = json.loads(response.text)
            break
        except Exception as e:
            print(e)
            continue

    overall_sentiment = 0 
    overall_quality = 0
    overall_output = ""
    for result, d in zip(results, input_data):
        result["text"] = d  # add the text to the result
        overall_sentiment += result["sentiment"]
        overall_quality += result["quality"]

        overall_output += f"{d}\n"
        
    overall_sentiment /= overall_quality
    overall_output += f"Overall sentiment for {query} in {len(results)}: {overall_sentiment}\n"

    return overall_sentiment, overall_output

if __name__ == "__main__":
    print(get_sentiment("$AAPL"))