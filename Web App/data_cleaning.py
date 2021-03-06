import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import nltk

from nltk import RegexpParser
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize, sent_tokenize 

from collections import Counter

from string import punctuation

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from web_scraper import commentScraper

# Convert comment list to dataframe
def createDataframe(username):
    
    comments = commentScraper(username)
    comments_df = pd.DataFrame(comments, columns=["posts"])

    #print(comments_df)
    return(comments_df)

def calculateModelParameters(username):

    #Call dataframe
    comments_df = createDataframe(username)

    #Number of words per comment
    comments_df["word_count"] = comments_df['posts'].str.split().apply(len)

    #Number of interrobangs per comment
    comments_df["interrobang_count"] = comments_df['posts'].apply(lambda x: x.count('?')) + comments_df['posts'].apply(lambda x: x.count('!'))

    #Average words per comment
    words_per_comment = comments_df["word_count"].mean()

    #Squared average words per comment
    squared_total_words = (words_per_comment)*2

    #Variance of words per comment
    word_count_variance_per_comment = comments_df.loc[:, "word_count"].var()

    #Average interrobangs per comment
    interrobangs_per_comment = comments_df["interrobang_count"].mean()

    #Import dataframe for counting nouns, adjectives, etc.
    tags_df = tokenizePosts(username)

    #Total number of nouns used
    tags_df["nouns"] = tags_df["Tagged Posts PosTag"].apply(NounCounter)
    tags_df["noun_count"] = tags_df["nouns"].str.len()
    noun_count = tags_df["noun_count"].sum()

    #Total number of adjectives used
    tags_df["adjectives"] = tags_df["Tagged Posts PosTag"].apply(AdjectiveCounter)
    tags_df["adjective_count"] = tags_df["adjectives"].str.len()
    adjective_count = tags_df["adjective_count"].sum()

    # Total number of verbs used
    tags_df["verbs"] = tags_df["Tagged Posts PosTag"].apply(VerbCounter)
    tags_df["verb_count"] = tags_df["verbs"].str.len()
    verb_count = tags_df["verb_count"].sum()

    # Total number of determiners used
    tags_df["determiners"] = tags_df["Tagged Posts PosTag"].apply(DeterminerCounter)
    tags_df["determiner_count"] = tags_df["determiners"].str.len()
    determiner_count = tags_df["determiner_count"].sum()

    # Total number of interjections used
    tags_df["interjections"] = tags_df["Tagged Posts PosTag"].apply(InterjectionCounter)
    tags_df["interjection_count"] = tags_df["interjections"].str.len()
    interjection_count = tags_df["interjection_count"].sum()

    # Total number of prepositions used
    tags_df["prepositions"] = tags_df["Tagged Posts PosTag"].apply(PrepositionCounter)
    tags_df["preposition_count"] = tags_df["prepositions"].str.len()
    preposition_count = tags_df["preposition_count"].sum()

    # Average sentiment score
    sentiment_score = sentimentScorer(username)

    #print(comments_df)
    # print(f'Average words per comment: {words_per_comment}')
    # print(f'Average words per comment squared: {squared_total_words}')
    # print(f'Word count variance per comment: {word_count_variance_per_comment}')
    # print(f'Average interrobangs per comment: {interrobangs_per_comment}')
    # print(f'Total nouns: {noun_count}')
    # print(f'Total adjectives: {adjective_count}')
    # print(f'Total verbs: {verb_count}')
    # print(f'Total determiners: {determiner_count}')
    # print(f'Total interjections: {interjection_count}')
    # print(f'Total prepositions: {preposition_count}')
    # print(f'Average sentiment score: {sentiment_score}')

    # Put parameters into dictionary
    model_parameters = {
        "words_per_comment": words_per_comment,
        "squared_total_words": squared_total_words,
        "word_count_variance_per_comment": word_count_variance_per_comment,
        "interrobangs_per_comment": interrobangs_per_comment,
        "noun_count": noun_count,
        "adjective_count": adjective_count,
        "verb_count": verb_count,
        "determiner_count": determiner_count,
        "interjection_count": interjection_count,
        "preposition_count": preposition_count,
        "sentiment_score": sentiment_score
    }

    return model_parameters

def tokenizePosts(username):

    comments_df = createDataframe(username)
    posts = comments_df["posts"]

    #remove URLs 
    posts_df = pd.DataFrame(data=posts)
    posts_df['posts'] = posts_df['posts'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
    posts_df['posts'] = posts_df['posts'].replace(r'\|\|\|', '', regex=True).replace(r'_____', '', regex=True).replace(r'@','', regex=True)


    posts_df['Tokenized Posts'] = posts_df.apply(lambda row: nltk.word_tokenize(row['posts']), axis=1)
    tokenized_df = pd.DataFrame(posts_df)

    #remove punctuations
    posts["Tokenized Posts"] = (strip_punctuation(str(tokenized_df["Tokenized Posts"])))

    #tokenization using postag 
    tokenized_df["Tagged Posts PosTag"] = posts_df.apply(lambda row: nltk.pos_tag(row["Tokenized Posts"]), axis=1)

    #assemble tags in lowercase dataframe with only Tagged 
    Tagged_Posts_PosTag = tokenized_df["Tagged Posts PosTag"]
    tags_df = pd.DataFrame(data = Tagged_Posts_PosTag)
    #str_tags_df = tags_df.astype(str)

    #print(tags_df)
    return(tags_df)

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

def NounCounter(tags_df):
    nouns = []
    for (word, pos) in tags_df:
        if pos.startswith("NN"):
            nouns.append(word)
    return nouns

def AdjectiveCounter(tags_df):
    adjectives = []
    for (word, pos) in tags_df:
        if pos.startswith("JJ"):
            adjectives.append(word)
    return adjectives

def VerbCounter(tags_df):
    verbs = []
    for (word, pos) in tags_df:
        if pos.startswith("V"):
            verbs.append(word)
    return verbs

def DeterminerCounter(tags_df):
    determiners = []
    for (word, pos) in tags_df:
        if pos.startswith("DT"):
            determiners.append(word)
    return determiners

def InterjectionCounter(tags_df):
    interjections = []
    for (word, pos) in tags_df:
        if pos.startswith("UH"):
            interjections.append(word)
    return interjections

def PrepositionCounter(tags_df):
    prepositions = []
    for (word, pos) in tags_df:
        if pos.startswith("IN"):
            prepositions.append(word)
    return prepositions

def sentimentScorer(username):
    analyzer = SentimentIntensityAnalyzer()
    comments_df = createDataframe(username)
    target_string = comments_df['posts']

    scores = []
    for i in target_string:
        compound = analyzer.polarity_scores(i)["compound"]
        scores.append(compound)
    
    comments_df['sentiment_score_compound'] = scores
    sentiment_score = comments_df['sentiment_score_compound'].mean()

    return sentiment_score

#username = input("Input username:")
#calculateModelParameters(username)
#tokenizePosts(username)