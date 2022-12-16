import string
import re
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import tensorflow as tf
# import tensorflow_recommenders as tfrs
from collections import Counter
from typing import Dict, Text
from ast import literal_eval
from datetime import datetime
# from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD, SVDpp, NMF, KNNBasic
from surprise.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn import svm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity





# Create a second dataset with the metadata added to the ratings
def create_meta_ratings():
    meta_ratings_df = pd.read_csv('ratings_small.csv')
    meta_ratings_df = meta_ratings_df.merge(df[['id', 'original_title', 'genres', 'overview', 'keywords']], left_on='movieId',right_on='id', how='left')
    meta_ratings_df = meta_ratings_df[~meta_ratings_df['id'].isna()] # remove entries where metadata does not exist
    meta_ratings_df.drop('id', axis=1, inplace=True)
    meta_ratings_df.drop('timestamp', axis=1, inplace=True)
    meta_ratings_df.reset_index(drop=True, inplace=True)
    meta_ratings_test, meta_ratings_train = train_test_split(meta_ratings_df)
    return meta_ratings_test, meta_ratings_train

# Load raw ratings file to perform Matrix Factorization on
def create_ratings(meta_ratings_train, meta_ratings_test):
    ratings_train, ratings_test = meta_ratings_train, meta_ratings_test
    for ratings_df in [ratings_train, ratings_test]:
        ratings_df.drop('original_title', axis=1, inplace=True)
        ratings_df.drop('genres', axis=1, inplace=True)
        ratings_df.drop('overview', axis=1, inplace=True)
        ratings_df.drop('keywords', axis=1, inplace=True)
        ratings_df.reset_index(drop=True, inplace=True)
       
    return ratings_train, ratings_test

def clean_keywords(row):
    x = row.keywords
    stored_words = []
    semi_count = 0
    prev_semiindex = 0
    for i in range(len(x)):
        element = x[i]
        if element == ':':
            semi_count += 1
            if semi_count%2 == 0:
                prev_semiindex = i
        if element == "}":
            stored_words.append(x[prev_semiindex+3:i-1])
    return stored_words

def clean_genres(row):
    x = row.genres
    stored_words = []
    semi_count = 0
    prev_semiindex = 0
    for i in range(len(x)):
        element = x[i]
        if element == ':':
            semi_count += 1
            if semi_count%2 == 0:
                prev_semiindex = i
        if element == "}":
            stored_words.append(x[prev_semiindex+3:i-1])
    return stored_words

# Comparison of sizes

def create_dataframes():
    meta_ratings_df = create_meta_ratings()
    meta_ratings_train, meta_ratings_test = meta_ratings_df
    ratings_df = create_ratings(meta_ratings_train.copy(), meta_ratings_test.copy())
    ratings_df_train, ratings_df_test = ratings_df
    for df in [meta_ratings_train, meta_ratings_test]:
        df['keywords'] = df.apply(clean_keywords, axis=1)
        df['genres'] = df.apply(clean_genres, axis=1)
    print('shapes')
    print(len(ratings_df_train), len(meta_ratings_train))
    print('this is the ratings dataset')
    print(ratings_df_train[0:10])
    print('Contains userID, movieID, and ratings')
    print(meta_ratings_train[0:10])
    return meta_ratings_train, meta_ratings_test, ratings_df_train, ratings_df_test


# Performing SVD (Netflix Prize Algorithm) on Ratings dataset
# This is our benchmark, uses collaborative filtering

def svd_eval(algo, train_dataset, test_dataset):
    reader = Reader()
    data = Dataset.load_from_df(train_dataset, reader)
    
    cross_validate(algo, data, measures=['RMSE', 'MAE'])
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    rmse = 0 
    for index, row in test_dataset.iterrows():
        error = (algo.predict(row.userId, row.movieId).est - row.rating)
        error *= error
        rmse += error
    rmse = rmse/len(test_dataset)
    return rmse

def make_vector(row, genre_dict):
  
    cat_encoding = np.zeros(len(genre_dict))
    for genre in row:
        cat_encoding[genre_dict[genre]] = 1
    
    return cat_encoding

def make_vector_sort(rows, genre_dict, gen):
    out = []
    cat_encoding = np.zeros(len(genre_dict))
    for row in rows:
        for genre in row:
            cat_encoding[genre_dict[genre]] = 1
    
    out.append(np.dot(cat_encoding, gen))
    return out


def genre_info(train_dataset, test_dataset):
    
    print(train_dataset['genres'])
    print(train_dataset.columns)
   
    genre_dict = dict()
    i = 0
    for row in train_dataset['genres']:
        for genre in row:
            if genre not in genre_dict:
                genre_dict[genre] = i
                i += 1
    
    for index, row in test_dataset.iterrows():
        
        user, movie, gen = row.userId, row.movieId, make_vector(row.genres, genre_dict)
        print('hi', gen)
        temp_df = train_dataset.loc[train_dataset['userId'] == user]
    
        temp_df['correlation'] = temp_df['genres'].apply(lambda y: np.dot(gen,make_vector(y, genre_dict)))
        temp_df.sort_values(by = 'correlation')
        print(temp_df['correlation'])
    

def genre_svm(train_dataset, test_dataset):
    print('hgelo,', train_dataset.columns)
    genre_vector = []
    genre_dict = dict()
    i = 0
    for row in train_dataset['genres']:
        for genre in row:
            if genre not in genre_dict:
                genre_dict[genre] = i
                i += 1

    rmse = 0
    for index, row in test_dataset.iterrows():
        genre_vector = []
        user, movie, gen, true_rating = row.userId, row.movieId, make_vector(row.genres, genre_dict), row.rating
        train_user = train_dataset.loc[train_dataset['userId'] == user]
        for row in train_user['genres']:
            cat_encoding = np.zeros(len(genre_dict))
            for genre in row:
                cat_encoding[genre_dict[genre]] = 1
            genre_vector.append(cat_encoding)
        df = pd.DataFrame(genre_vector)
        
        svr = svm.SVR(kernel='poly')
        svr.fit(df, train_user['rating'])

        error = true_rating - svr.predict([gen])
        error *= error
        rmse += error

    rmse = rmse/len(test_dataset)
    print(rmse)
    return rmse[0]
    
def title(df, index):
    return df[df.index == index]['original_title'].values

def index(df, title):
    return df[df.original_title == title]['index'].values[0]

def bert_similarity(train, test):
    train.reset_index(drop=True, inplace=True)
    df = train
    print(df['original_title'].head())

    df['index'] = [i for i in range(len(df))]
    bert = SentenceTransformer('all-MiniLM-L12-v2')
    sentence_embeddings = bert.encode(df['overview'].tolist())
    similarity = cosine_similarity(sentence_embeddings)
    #print(similarity)
    rec = sorted(list(enumerate(similarity[index(df, 'Straight From the Heart')])), key = lambda x:x[1], reverse = True)
    print(rec)
    for i in range(5):
        print(title(df, rec[i][0]))


if __name__ == '__main__':
    # Load Metadata files (keywords and Metadata and preprocess)
    #credits = pd.read_csv('the-movies-dataset/credits.csv')
    keywords = pd.read_csv('keywords.csv')
    movies = pd.read_csv('movies_metadata.csv', low_memory=False).drop(['belongs_to_collection', 'homepage', 'imdb_id', 'poster_path', 'status', 'title', 'video'], axis=1).drop([19730, 29503, 35587]) # Incorrect data type
    movies['id'] = movies['id'].astype('int64')
    df = movies.merge(keywords, on='id')
    df['original_language'] = df['original_language'].fillna('')
    df['runtime'] = df['runtime'].fillna(0)
    df['tagline'] = df['tagline'].fillna('')
    df.dropna(inplace=True)

    meta_ratings_train, meta_ratings_test, ratings_df_train, ratings_df_test = create_dataframes()

    '''svd = SVD()
    svdpp = SVDpp()
    knn = KNNBasic()
    nmf = NMF()

    print('*' * 50)

    rmse_svd = svd_eval(svd, ratings_df_train, ratings_df_test)
    rmse_svd_plus = svd_eval(svdpp, ratings_df_train, ratings_df_test)
    rmse_nmf = svd_eval(nmf, ratings_df_train, ratings_df_test)
    knn = svd_eval(knn, ratings_df_train, ratings_df_test)

    print('*' * 50)

    print('SVD', rmse_svd)
    print('SVD+', rmse_svd_plus)
    print('NMF', rmse_nmf)
    print('KNN', knn)

    print('*' * 50)'''

    # genre_info(meta_ratings_train, meta_ratings_test)

    #genre_svm(meta_ratings_train, meta_ratings_test)

    bert_similarity(meta_ratings_train, meta_ratings_test)

    
