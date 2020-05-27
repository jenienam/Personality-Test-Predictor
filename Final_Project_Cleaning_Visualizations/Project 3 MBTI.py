#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install textblob


# In[2]:


import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import nltk

from nltk import RegexpParser
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 

stop_words = set(stopwords.words('english')) 

from collections import Counter

import textblob
from textblob import TextBlob, Word, Blobber
from textblob.classifiers import NaiveBayesClassifier
from textblob.taggers import NLTKTagger

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from bs4 import BeautifulSoup


# In[3]:


mbti_df = pd.read_csv('Resources/mbti_1.csv')
mbti_df.head()


# In[4]:


# clean text in 'posts' using Beautiful Soup
#def cleaning(text):
#    text = BeautifulSoup(text, "lxml").text
#    text = re.sub(r'\|\|\|', r' ', text) 
#    text = re.sub(r'http\S+', r'<URL>', text)
#    return text


# In[5]:


#Apply Beautiful Soup to the mbti_df
#mbti_df['cleaned_posts'] = mbti_df['posts'].apply(cleaning)
#mbti_df
#df = mbti_df.drop(columns="posts")


# In[6]:


#count the different mbti
mbti_counts = mbti_df['type'].value_counts()
mbti_counts.head()


# In[7]:


#for where a new comment begins
def var_row(row):
    l = []
    for i in row.split('|||'):
        l.append(len(i.split()))
    return np.var(l)


# In[8]:


#define mbti and add the description to the type to the chart
mbti = {'I':'Introvert', 'E':'Extrovert', 'N':'Intuition', 
        'S':'Sensing', 'T':'Thinking', 'F': 'Feeling', 
        'J':'Judging', 'P': 'Perceiving'}
#description of the type 
mbti_df['description'] = mbti_df['type'].apply(lambda x: ' '.join([mbti[l] for l in list(x)]))
#words per comment
mbti_df['average_words_per_comment'] = mbti_df['posts'].apply(lambda x: len(x.split())/50)
#squared totals
mbti_df['average_squared_total_words'] = mbti_df['average_words_per_comment']*2
#word count varianc
mbti_df['average_word_count_variance_per_comment'] = mbti_df['posts'].apply(lambda x: var_row(x))
#interrobangs per comment = 
mbti_df['average_interrobangs_per_comment']=mbti_df['posts'].apply(lambda x: x.count('?')/50) + mbti_df['posts'].apply(lambda x: x.count('!')/50)
#preview
mbti_df.head()


# In[9]:


#tokenize to find pars of speeches and pre-processing

posts = mbti_df["posts"]

#remove URLs 
posts_df = pd.DataFrame(data=posts)
posts_df['posts'] = posts_df['posts'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
posts_df['posts'] = posts_df['posts'].replace(r'\|\|\|', '', regex=True).replace(r'_____', '', regex=True).replace(r'@','', regex=True)


posts_df['Tokenized Posts'] = posts_df.apply(lambda row: nltk.word_tokenize(row['posts']), axis=1)
tokenized_df = pd.DataFrame(posts_df)
tokenized_df

#remove punctuations
from string import punctuation
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)
posts["Tokenized Posts"] = (strip_punctuation(str(tokenized_df["Tokenized Posts"])))


# In[10]:


#tokenization using postag 
tokenized_df["Tagged Posts PosTag"] = posts_df.apply(lambda row: nltk.pos_tag(row["Tokenized Posts"]), axis=1)
tokenized_df


# alternative method for word tagging below 

# In[11]:


#tagged tokenization using stopwords & textblob
#tokenized = sent_tokenize(str(tokenized_df["Tokenized Posts"])) 
#for i in tokenized: 
    
#    wordsList = nltk.word_tokenize(i) 
  
    # removing stop words from wordList 
#    wordsList = [w for w in wordsList if not w in stop_words]  
  
    #  Using a Tagger. Which is part-of-speech  
    # tagger or POS-tagger.  
#    tagged = nltk.pos_tag(wordsList) 
  
#    tokenized_df["Tagged Posts Stopwords"] = posts_df.apply(lambda row: nltk.pos_tag(wordsList), axis=1)

#textblob


# In[12]:


#assemble tags in lowercase dataframe with only Tagged 
Tagged_Posts_PosTag = tokenized_df["Tagged Posts PosTag"]
tags_df = pd.DataFrame(data = Tagged_Posts_PosTag)
str_tags_df = tags_df.astype(str)
#nouns
def NounCounter(tags_df):
    nouns = []
    for (word, pos) in tags_df:
        if pos.startswith("NN"):
            nouns.append(word)
    return nouns
tags_df["type"] = mbti_df["type"]
tags_df["description"] = mbti_df["description"]
tags_df["nouns"] = tags_df["Tagged Posts PosTag"].apply(NounCounter)
tags_df["noun_count"] = tags_df["nouns"].str.len()
tags_df


# In[13]:


#adjectives
def AdjectiveCounter(tags_df):
    adjectives = []
    for (word, pos) in tags_df:
        if pos.startswith("JJ"):
            adjectives.append(word)
    return adjectives
tags_df["adjectives"] = tags_df["Tagged Posts PosTag"].apply(AdjectiveCounter)
tags_df["adjectives_count"] = tags_df["adjectives"].str.len()

tags_df


# In[14]:


#verbs
def VerbCounter(tags_df):
    verbs = []
    for (word, pos) in tags_df:
        if pos.startswith("V"):
            verbs.append(word)
    return verbs
tags_df["verbs"] = tags_df["Tagged Posts PosTag"].apply(VerbCounter)
tags_df["verb_count"] = tags_df["verbs"].str.len()
tags_df


# In[15]:


#Determiners
def DeterminerCounter(tags_df):
    determiners = []
    for (word, pos) in tags_df:
        if pos.startswith("DT"):
            determiners.append(word)
    return determiners
tags_df["determiners"] = tags_df["Tagged Posts PosTag"].apply(DeterminerCounter)
tags_df["determiner_count"] = tags_df["determiners"].str.len()
tags_df


# In[16]:


#interjections
def InterjectionCounter(tags_df):
    interjections = []
    for (word, pos) in tags_df:
        if pos.startswith("UH"):
            interjections.append(word)
    return interjections
tags_df["interjections"] = tags_df["Tagged Posts PosTag"].apply(InterjectionCounter)
tags_df["interjections_count"] = tags_df["interjections"].str.len()
tags_df


# In[17]:


#prepositions
def PrepositionCounter(tags_df):
    prepositions = []
    for (word, pos) in tags_df:
        if pos.startswith("IN"):
            prepositions.append(word)
    return prepositions
tags_df["prepositions"] = tags_df["Tagged Posts PosTag"].apply(PrepositionCounter)
tags_df["preposition_count"] = tags_df["prepositions"].str.len()
tags_df = tags_df.rename(columns={'adjectives_count': 'adjective_count', 'interjections_count': 'interjection_count'})
tags_df


# In[18]:


#parts of speech data frame
types = tags_df["type"]
parts_of_speech_df = pd.DataFrame(data = types)
parts_of_speech_df["noun_count"] = tags_df["noun_count"]
parts_of_speech_df["adjective_count"] = tags_df["adjective_count"]
parts_of_speech_df["verb_count"] = tags_df["verb_count"]
parts_of_speech_df["determiner_count"] = tags_df["determiner_count"]
parts_of_speech_df["interjection_count"] = tags_df["interjection_count"]
parts_of_speech_df["preposition_count"] = tags_df["preposition_count"]


# In[19]:


parts_of_speech_df.to_csv("parts_of_speech.csv")
tags_df.to_csv("mbti_data_full.csv")
tokenized_df.to_csv("tokenized_data.csv")


# In[23]:





# VISUALIZATIONS

# In[20]:


#see the different types of mbti and create a bar graph of counts 
plt.figure(figsize=(12,4))
count = mbti_df['type'].value_counts()
df_counts = sns.barplot(count.index, count.values, data=mbti_df)
plt.ylabel('Total Count', fontsize=12)
plt.xlabel('Personality Types', fontsize=12)


# In[21]:


#comparison bar charts based on personality types 
#created new data frame with types 
split_types = mbti_df["description"].str.split(" ")
type_data = split_types.tolist()
types = ["I_vs_E", "N_vs_S", "F_vs_T", "J_vs_P"]
new_type_data = pd.DataFrame(type_data, columns=types)
new_type_data.head()


# In[22]:


#bar charts for individual personality types 
sns.catplot(x=new_type_data["I_vs_E"], kind="count",
    palette="pastel", edgecolor=".6", data=new_type_data);
#sns.catplot(x=new_type_data["N_vs_S"], kind="count",
#    palette="pastel", edgecolor=".6", data= new_type_data);
#sns.catplot(x=new_type_data["F_vs_T"], kind="count",
#    palette="pastel", edgecolor=".6", data= new_type_data);
#sns.catplot(x=new_type_data["J_vs_P"], kind="count",
#    palette="pastel", edgecolor=".6", data= new_type_data);


# In[ ]:


#words per chart
plt.figure(figsize=(12,4))
sns.swarmplot(x= mbti_df["type"], y=mbti_df["words_per_comment"]) # Set color paletteplt.ylabel('Total Count', fontsize=12)
plt.xlabel('Personality Types', fontsize=12)
plt.ylabel('Words per Comment', fontsize=12)


# In[ ]:


#interrobangs_per_comment
plt.figure(figsize=(12,4))
sns.boxplot(x= mbti_df["type"], y=mbti_df["interrobangs_per_comment"]) 
plt.xlabel('Personality Types', fontsize=12)
plt.ylabel('Interrobangs per Comment', fontsize=12)


# In[ ]:




