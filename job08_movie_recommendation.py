import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from konlpy.tag import Okt
import re
from gensim.models import Word2Vec

def getRecommendation(cosin_sim):
    simScore = list(enumerate(cosin_sim[-1]))
    simScore = sorted(simScore, key=lambda x:x[1], reverse=True)
    simScore = simScore[:11]
    movieIdx = [i[0] for i in simScore]
    recMovieList = df_reviews.iloc[movieIdx, 0]
    return recMovieList


df_reviews = pd.read_csv('./crawling_data/reviews_2017_2022.csv')
Tfidf_matrix = mmread('./models/Tfidf_movie_review.mtx').tocsr()
with open('./models/tfidf.pickle', 'rb') as f:
    Tfidf = pickle.load(f)

# 영화 제목 / index를 이용
# movie_idx = df_reviews[df_reviews['titles']=='어벤져스: 엔드게임 (Avengers: Endgame)'].index[0]
# # movie_idx = 1003
# cosine_sim = linear_kernel(Tfidf_matrix[movie_idx], Tfidf_matrix)
# recommendation = getRecommendation(cosine_sim)
# print(recommendation[1:11])

# keyword 이용
# embedding_model = Word2Vec.load('./models/word2vec_2017_2020_movies.model')
# keyword = '마블'
# sim_word = embedding_model.wv.most_similar(keyword, topn=10)
# words = [keyword]
# for word, _ in sim_word:
#     words.append(word)
# sentence = []
# count = 10
# for word in words:
#     sentence = sentence + [word] * count
#     count -= 1
# sentence = ' '.join(sentence)
# sentence_vec = Tfidf.transform([sentence])
# cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
# recommendation = getRecommendation(cosine_sim)
# print(recommendation)

# 문장 이용
okt = Okt()
sentence = '견딜 수 없이 촌스런 삼남매의 견딜 수 없이 사랑스러운 행복소생기'
review = re.sub('[^가-힣 ]', ' ', sentence)

token = okt.pos(review, stem=True)

df_token = pd.DataFrame(token, columns=['word', 'class'])
df_token = df_token[(df_token['class'] == 'Noun') |
                    (df_token['class'] == 'Verb') |
                    (df_token['class'] == 'Adjective')]
words = []
for word in df_token.word:

    if 1 < len(word):
        words.append(word)
cleaned_sentence = ' '.join(words)
print(cleaned_sentence)
sentence_vec = Tfidf.transform([cleaned_sentence])
cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
recommendation = getRecommendation(cosine_sim)
print(recommendation)


