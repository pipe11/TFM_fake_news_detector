#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
start = time.time()

import itertools
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords  
from nltk import word_tokenize  
from nltk.data import load  
from nltk.stem import SnowballStemmer  
from string import punctuation

from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import gc

####### Feature text extraction #######

def extract_features(text):
    
    df = pd.read_csv('/home/pipe11/TFM_fake_news_detector/data/corpus_spanish.csv')

    # Label encoder
    labelencoder = LabelEncoder()
    df['Labels'] = labelencoder.fit_transform(df['Category'])
    y = df['Labels']

    # index for later sparse Matrix
    traindex = pd.RangeIndex(start = 0, stop = 971, step = 1)
    predictindex = pd.RangeIndex(start = 0, stop = 972, step = 1)

    df_text = pd.DataFrame([[text]], columns = ['Text'])
    
    # concat the new to predict at the end of the df
    df_corpus = pd.concat([df[['Text']], df_text], axis = 0)

    df_corpus.reset_index(drop=True, inplace=True)
    
    df_features = pd.DataFrame()

    n_sentences = []
    n_words = []
    avg_words_sent = []
    avg_word_size = []
    type_token_ratio = []
    list_text = []

    for i, row in df_corpus.iterrows():
        text = df_corpus['Text'].iloc[i]

        text = text.replace(r"http\S+", "")
        text = text.replace(r"http", "")
        text = text.replace(r"@\S+", "")
        text = text.replace(r"(?<!\n)\n(?!\n)", " ")
        text = text.lower()

        sent_tokens = nltk.sent_tokenize(text)

        #Number of sentences
        number_sentences = len(sent_tokens)

        word_tokens = nltk.word_tokenize(text)

        stop_words = stopwords.words('spanish')
        stop_words.extend(list(punctuation))
        stop_words.extend(['¿', '¡', '"', '``']) 
        stop_words.extend(map(str,range(10)))

        filtered_tokens = [i for i in word_tokens if i not in stop_words]

        #number of tokens
        number_words = len(filtered_tokens)

        # average words per sentence
        avg_word_sentences = (float(number_words)/number_sentences)

        # average word size
        word_size = sum(len(word) for word in filtered_tokens) / number_words

        # type token ratio
        types = nltk.Counter(filtered_tokens)
        TTR = (len(types) / number_words) * 100

        n_sentences.append(number_sentences)
        n_words.append(number_words)
        avg_words_sent.append(avg_word_sentences)
        avg_word_size.append(word_size)
        type_token_ratio.append(TTR)
        list_text.append(text)

    df_features['sentences'] = n_sentences
    df_features['n_words'] = n_words
    df_features['avg_words_sent'] = avg_words_sent
    df_features['avg_word_size'] = avg_word_size
    df_features['TTR'] = type_token_ratio
    df_features['Text'] = list_text

    return df_features, y, traindex, predictindex

####### TFIDF Transformation to text ########

#Stopword list to use
spanish_stopwords = stopwords.words('spanish')

#Spanish stemmer:
stemmer = SnowballStemmer('spanish')

def stem_tokens(tokens, stemmer):  
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

#Punctuation to remove
non_words = list(punctuation)

#Adding spanish punctuation
non_words.extend(['¿', '¡'])  
non_words.extend(map(str,range(10)))

def tokenize(text):  
    #Remove punctuation
    text = ''.join([c for c in text if c not in non_words])
    #Tokenize
    tokens =  word_tokenize(text)

    #Stem
    try:
        stems = stem_tokens(tokens, stemmer)
    except Exception as e:
        print(e)
        print(text)
        stems = ['']
    return stems

def tfidf_transformer(df_features):

    tfidf_vectorizer = TfidfVectorizer(  
                    analyzer = 'word',
                    tokenizer = tokenize,
                    lowercase = True,
                    stop_words = spanish_stopwords)

    text_vectorized = tfidf_vectorizer.fit_transform(df_features['Text'])

    return text_vectorized

####### Combine TF-IDF and dense features #######

def feature_combiner(text_vectorized, df_features, traindex):

    categorical_features = ['sentences', 'n_words', 'avg_words_sent', 'avg_word_size', 'TTR']

    X = hstack([csr_matrix(df_features[categorical_features].loc[traindex, :].values), text_vectorized[0: traindex.shape[0]]])
    X_text = hstack([csr_matrix(df_features[categorical_features].loc[[traindex.shape[0]], :].values), text_vectorized[traindex.shape[0]:]])

    gc.collect();

    return X, X_text

####### Model training #######

def model_predictor(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 43)


    model_rf = RandomForestClassifier(bootstrap = True, max_depth = 95, max_features = 'auto', min_samples_leaf = 1, 
                            min_samples_split = 4, n_estimators = 1800)

    model_rf.fit(X_train, y_train)
    
    return model_rf

####### Outer function #######

def fake_news_detector():
    text = input('Paste the text content of a new: ')
    df_features, y, traindex, predictindex = extract_features(text)
    text_vectorized = tfidf_transformer(df_features)
    X, X_text = feature_combiner(text_vectorized, df_features, traindex)
    model_rf = model_predictor(X, y)
    
    if model_rf.predict(X_text) == 0:
        return print('This new smells fake!')
    else:
        return print('This new is totally true!')

