
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import spacy
from nltk import FreqDist
from lexical_diversity import lex_div as ld

pd.options.display.max_columns = None


#### text features ####

@st.cache(show_spinner = False)
def get_news_features(headline, text):
    nlp = spacy.load('es_core_news_lg')
        
    ## headline ##
    headline = headline.replace(r"http\S+", "")
    headline = headline.replace(r"http", "")
    headline = headline.replace(r"@\S+", "")
    headline = headline.replace(r"(?<!\n)\n(?!\n)", " ")
    headline = headline.lower()
    doc_h = nlp(headline)

    list_tokens_h = []
    list_tags_h = []
    n_sents_h = 0

    for sentence_h in doc_h.sents:
        n_sents_h += 1
        for token in sentence_h:
            list_tokens_h.append(token.text)

    fdist_h = FreqDist(list_tokens_h)

    # headline complexity features
    n_words_h = len(list_tokens_h)
    word_size_h = sum(len(word) for word in list_tokens_h) / n_words_h
    unique_words_h = (len(fdist_h.hapaxes()) / n_words_h) * 100
    ttr_h = ld.ttr(list_tokens_h) * 100
    mltd_h = ld.mtld(list_tokens_h)



    ## text content ##   
    text = text.replace(r"http\S+", "")
    text = text.replace(r"http", "")
    text = text.replace(r"@\S+", "")
    text = text.replace(r"(?<!\n)\n(?!\n)", " ")
    text = text.lower()
    doc = nlp(text)

    list_tokens = []
    list_pos = []
    list_tag = []
    n_sents = 0

    for sentence in doc.sents:
        n_sents += 1
        for token in sentence:
            list_tokens.append(token.text)
            list_pos.append(token.pos_)
            list_tag.append(token.tag_)

    n_pos = nltk.Counter(list_pos)
    n_tag = nltk.Counter(list_tag)
    fdist = FreqDist(list_tokens)

    # complexity features
    n_words = len(list_tokens)
    avg_word_sentences = (float(n_words) / n_sents)
    word_size = sum(len(word) for word in list_tokens) / n_words
    unique_words = (len(fdist.hapaxes()) / n_words) * 100
    ttr = ld.ttr(list_tokens) * 100
    mltd = ld.mtld(list_tokens)

    # lexical features
    n_quotes = n_tag['PUNCT__PunctType=Quot']
    quotes_ratio = (n_quotes / n_words) * 100
    propn_ratio = (n_pos['PROPN'] / n_words) * 100 
    noun_ratio = (n_pos['NOUN'] / n_words) * 100 
    adp_ratio = (n_pos['ADP'] / n_words) * 100
    det_ratio = (n_pos['DET'] / n_words) * 100
    punct_ratio = (n_pos['PUNCT'] / n_words) * 100 
    pron_ratio = (n_pos['PRON'] / n_words) * 100
    verb_ratio = (n_pos['VERB'] / n_words) * 100
    adv_ratio = (n_pos['ADV'] / n_words) * 100
    sym_ratio = (n_tag['SYM'] / n_words) * 100

    # create df

    df_features = pd.DataFrame({'n_sents': [n_sents], 'n_words': [n_words], 'avg_words_sents': [avg_word_sentences], 
                'word_size': [word_size], 'unique_words': [unique_words], 'ttr': [ttr], 'mltd': [mltd], 'n_words_h': [n_words_h],
                'word_size_h': [word_size_h], 'unique_words_h': [unique_words_h], 'mltd_h': [mltd_h], 'n_quotes': [n_quotes],
                'quotes_ratio': [quotes_ratio], 'propn_ratio': [propn_ratio], 'noun_ratio': [noun_ratio], 'adp_ratio': [adp_ratio],
                'det_ratio': [det_ratio], 'punct_ratio': [punct_ratio], 'pron_ratio': [pron_ratio], 'verb_ratio': [verb_ratio],
                'adv_ratio': [adv_ratio], 'sym_ratio': [sym_ratio]})
    
    return df_features


#### predictions ####

@st.cache(show_spinner = False)
def get_predictions(pickle_file, df_features):
    with st.spinner("Detecting..."):
        import pickle
        model = pickle.load(open(pickle_file, 'rb'))

        # prediction
        prediction = (model.predict(df_features)[0])
        proba_fake = (model.predict_proba(df_features)[0][0])*100
        proba_legitimate = (model.predict_proba(df_features)[0][1])*100
    
    return prediction, proba_fake, proba_legitimate

#### streamlit configuration ####

# load pickle file 
pickle_file = './fake_news_predictorv2.pkl'

# page configuration
page_title = 'Fake News Detector'
layout = 'wide'
initial_sidebar_state = 'expanded'

# display title and description
st.title("Fake news predictor in spanish")
"""
Master Thesis in Data Science - KSchool

This app is powered with a Machine Learning Algorithm to detect which news are fake or legitimate in spanish language 

Insert the new's headline and new's content text, and lets detect!
"""
# text input for headline and new's content
headline = st.text_input("Insert the headline text:")
text = st.text_input("Insert the new's text:")

## run functions##
if (text != "") & (headline != ""):
    
    df_features = get_news_features(headline, text)
    prediction, proba_fake, proba_legitimate = get_predictions(pickle_file, df_features)


    if prediction == 0:
        st.write('This is a fake new, with a probability of', proba_fake)

    else:
        st.write('This is a legitimate new, with a probability of', proba_legitimate)
