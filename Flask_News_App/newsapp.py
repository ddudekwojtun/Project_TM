import pprint
import requests
from newsapi import NewsApiClient
from wordcloud import WordCloud
import matplotlib.pyplot as plt

api = NewsApiClient(api_key='2806c221a7ab44409a50ee02da3bc4f9')

api_key: str = "2806c221a7ab44409a50ee02da3bc4f9"

# Endpoint:
url = 'https://newsapi.org/v2/everything?'

# Query and number of returns:
parameters = dict(apiKey=api_key, q='climate', pageSize=20, sources='bbc-news')

# Make the request
response = requests.get(url, params=parameters)

# converting to JSON:
response_json = response.json()
pprint.pprint(response_json)

for i in response_json['articles']:
    print(i['title'])


text_combined = ''
# Loop through all the headlines and add them to 'text_combined'
for i in response_json['articles']:
    text_combined += i['title'] + ' ' # add a space after every headline
print(text_combined[0:300])

wordcloud = WordCloud(max_font_size=40).generate(text_combined)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()




