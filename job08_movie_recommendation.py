import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle

def getRecommendation(cosin_sim):
    simScore = list(enumerate(cosin_sim[-1]))
    simScore = sorted(simScore, key=lambda x:x[1], reverse=True)
    simScore = simScore[1:11]
    movieIdx = [i[0] for i in simScore]
    recMovieList = df_reviews.iloc[movieIdx, 0]
    return recMovieList


df_reviews = pd.read_csv('./crawling_data/reviews_2017_2022.csv')
Tfidf_matrix = mmread('./models/Tfidf_movie_review.mtx').tocsr()
with open('./models/tfidf.pickle', 'rb') as f:
    Tfidf = pickle.load(f)

# 열화 제목 / index를 이용
movie_idx = df_reviews[df_reviews['titles']=='어벤져스: 엔드게임 (Avengers: Endgame)'].index[0]
# movie_idx = 1003
cosine_sim = linear_kernel(Tfidf_matrix[movie_idx], Tfidf_matrix)
recommendation = getRecommendation(cosine_sim)
print(recommendation)