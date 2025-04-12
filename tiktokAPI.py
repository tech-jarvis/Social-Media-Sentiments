# =======================================================
# COMMENTS EXTRACTION FROM TIKTOK BASED ON VIDEO ID
# =======================================================


# Before running this code, make sure to run the following two commands:
# python -m playwright install
# playwright install



# %pip install TikTokApi
import asyncio
import os
import pandas as pd
from TikTokApi import TikTokApi
import re



def extract_tiktok_video_id(url_string):
    # Define the regular expression pattern to match TikTok URLs
    pattern = r'tiktok\.com.*\/video\/(\d+)'
    
    # Use re.search to find the first match of the pattern in the URL string
    match = re.search(pattern, url_string)
    
    # If a match is found and there's a captured group (video ID), return it
    if match and match.group(1):
        return match.group(1)
    
    # If no match is found or there's no captured group, return None
    return None


tiktok_url = "https://www.tiktok.com/@shagor_512/video/7274560944716467457?is_from_webapp=1&sender_device=pc"


video_id = extract_tiktok_video_id(tiktok_url)
if video_id:
    print("TikTok Video ID:", video_id)






# video_id = 7268213547245636869
#video_id =7235112760009362693
ms_token = os.environ.get("ms_token", None)  # set your own ms_token

async def get_comments():
    comments_list = []
    
    async with TikTokApi() as api:
        await api.create_sessions(ms_tokens=[ms_token], num_sessions=1, sleep_after=3)
        video = api.video(id=video_id)
        
        async for comment in video.comments(count=100):
            comments_list.append(comment.as_dict)
    
    return comments_list

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    comments = loop.run_until_complete(get_comments())
    # df = pd.DataFrame(comments)
    # df.to_csv("Comments.csv", index=True)
    comment_texts = [comment['text'] for comment in comments]
    df_comments = pd.DataFrame({'text': comment_texts})
    df_comments.to_csv("Justcomments.csv", index=True)


# =======================================================
# LOADING THE MODEL AND SENTIMENT ANALYSIS
# =======================================================


import requests
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from scipy.special import softmax


MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment-latest'


# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Function to analyze sentiment
def analyze_sentiment(text):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    output = model(**encoded_input)
    scores = output.logits[0].detach().numpy()
    scores = softmax(scores, axis=0)

    # Labels corresponding to sentiment classes
    labels = ["negative", "neutral", "positive"]

    json_sent = {
        "label": labels[np.argmax(scores)],
        "probability": {
            "neg": scores[0],  # Negative sentiment score
            "neutral": scores[1],  # Neutral sentiment score
            "pos": scores[2]  # Positive sentiment score
        }
    }

    return json_sent


import pandas as pd
import numpy as np
 
# Assuming you already have 'new_df' with the comments

# Create a new DataFrame to store sentiment analysis results
result_df = pd.DataFrame(columns=['textDisplay', 'label', 'pos', 'neg', 'neutral'])

# Iterate through each row in the original DataFrame
for index, row in df_comments.iterrows():
    comment = row['text']
    
    # Analyze sentiment for the comment using the analyze_sentiment function
    sentiment_result = analyze_sentiment(comment)
    
    # Add sentiment analysis results to the new DataFrame
    result_df.loc[index] = [
        comment,
        sentiment_result['label'],
        sentiment_result['probability']['pos'],
        sentiment_result['probability']['neg'],
        sentiment_result['probability']['neutral']
    ]

# Print the resulting DataFrame with sentiment analysis results
print(result_df.head())  # Print the first 5 entries

# Save the DataFrame to a CSV file
result_df.to_csv('sentiment_analysis_results.csv', index=False)




# =======================================================
# Calculating the overall sentiment of the comment section
# =======================================================

# Calculate weighted average for pos, neg, and neutral columns
weighted_avg_pos = result_df['pos'].mean()
print("weighted_avg_pos ", weighted_avg_pos)
weighted_avg_neg = result_df['neg'].mean()
print("weighted_avg_neg ", weighted_avg_neg)

weighted_avg_neutral = result_df['neutral'].mean()
print("weighted_avg_neutral ", weighted_avg_neutral)


# Determine the label based on the greatest weighted average
max_weighted_avg = max(weighted_avg_pos, weighted_avg_neg, weighted_avg_neutral)
print("max_weighted_avg ",max_weighted_avg)
label = None
if max_weighted_avg == weighted_avg_pos:
    label = 'positive'
elif max_weighted_avg == weighted_avg_neg:
    label = 'negative'
else:
    label = 'neutral'

# Print the determined label
print(f"The Overall determined sentiment label is: {label}")


# =======================================================
# Calculating the absolute number of positive/negative/neutral comments
# =======================================================

# Calculate the total number of comments
total_comments = len(result_df)

# Count the number of positive, negative, and neutral comments
num_positive_comments = result_df[result_df['label'] == 'positive']['textDisplay'].count()
num_negative_comments = result_df[result_df['label'] == 'negative']['textDisplay'].count()
num_neutral_comments = result_df[result_df['label'] == 'neutral']['textDisplay'].count()

# Print the results
print(f"Total Comments: {total_comments}")
print(f"Number of Positive Comments out of {total_comments} = {num_positive_comments}")
print(f"Number of Negative Comments out of {total_comments} = {num_negative_comments}")
print(f"Number of Neutral Comments out of {total_comments} = {num_neutral_comments}")


# =======================================================
# Calculating the percentage of positive/negative/neutral comments
# =======================================================

percentage_positive = (num_positive_comments / total_comments) * 100
percentage_negative = (num_negative_comments / total_comments) * 100
percentage_neutral = (num_neutral_comments / total_comments) * 100

# Print the results
print(f"Percentage of Positive Comments: {percentage_positive:.2f}%")
print(f"Percentage of Negative Comments: {percentage_negative:.2f}%")
print(f"Percentage of Neutral Comments: {percentage_neutral:.2f}%")


# =======================================================
# The most common words that appeared in the positive comments
# =======================================================

from nltk import FreqDist
import operator

import re
#the words that appear he most in positive reviews
import nltk
porter = nltk.PorterStemmer()
list_pos=[]
for i in range(len(result_df.loc[result_df['label'] == 'positive'])):
    list_pos.append(result_df.loc[result_df['label'] == 'positive']["textDisplay"].iloc[i])
lst_words_pos = []
for line in list_pos:
    text_pos = re.split('\n| |\?|\!|\:|\"|\(|\)|\...|\;',line)
    for word in text_pos:
        if (len(word)>3 and not word.startswith('@') and not word.startswith('#') and word != 'RT'):
            lst_words_pos.append(porter.stem(word.lower()))


dist_pos = FreqDist(lst_words_pos) 
sorted_dist_pos = sorted(dist_pos.items(), key=operator.itemgetter(1), reverse=True)
sorted_dist_pos[:50]



# =======================================================
# A list of the common words that appeared in the negative comments
# =======================================================

list_neg=[]
for i in range(len(result_df.loc[result_df['label'] == 'negative'])):
    list_neg.append(result_df.loc[result_df['label'] == 'negative']["textDisplay"].iloc[i])
lst_words_neg = []
for line in list_neg:
    text_neg = re.split('\n| |\?|\!|\:|\"|\(|\)|\...|\;',line)
    for word in text_neg:
        if (len(word)>3 and not word.startswith('@') and not word.startswith('#') and word != 'RT'):
            lst_words_neg.append(porter.stem(word.lower()))
dist_neg = FreqDist(lst_words_neg) 
sorted_dist_neg = sorted(dist_neg.items(), key=operator.itemgetter(1), reverse=True)
sorted_dist_neg[:50]