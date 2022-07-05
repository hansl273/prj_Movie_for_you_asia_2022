from unittest.mock import inplace

import pandas as pd
import stopwords as stopwords
from konlpy.tag import Okt
import re

df = pd.read_csv('./crawling_data/reviews_2020.csv')
df.info()

okt = Okt()

df_stopwords = pd.read_csv('./crawling_data/stopwords.csv')
stopwords = list(df_stopwords['stopword'])


cleaned_sentences = []
for review in df.reviews:
    review = re.sub('[^가-힣 ]', ' ', review)
    token = okt.pos(review, stem=True)

    df_token = pd.DataFrame(token, columns=['word', 'class'])
    df_token = df_token[(df_token['class'] == 'Noun') |
                        (df_token['class'] == 'Verb') |
                        (df_token['class'] == 'Adjective')]
    words = []
    for word in df_token.word:
        if len(word) > 1:
            if word not in stopwords:
                words.append(word)
    cleaned_sentence = ' '.join(words)
    cleaned_sentences.append(cleaned_sentence)

df['cleaned_sentences'] = cleaned_sentences
df = df[['title', 'cleaned_sentences']]
df.dropna(inplace=True)

df.to_csv('./crawling_data/cleaned_review_2020.csv', index=False)
df.info()

