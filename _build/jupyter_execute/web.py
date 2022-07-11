#!/usr/bin/env python
# coding: utf-8

# # UAS PPW

# ### Crawling

# In[1]:


import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        urls = [
            "https://pta.trunojoyo.ac.id/welcome/detail/080211100070",
 
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        yield{
        'Abstrak' : response.css('#content_journal > ul > li > div:nth-child(4) > div:nth-child(2) >  ::text').extract(),


        }


# ### Import yang di perlukan

# In[2]:


# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

#preprocessing
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

# for named entity recognition (NER)
from nltk import ne_chunk

# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#stop-words
stop_words=set(nltk.corpus.stopwords.words('english'))


# 

# In[9]:


df=pd.read_csv('jurnal.csv', usecols =['Abstrak_indo'])


# In[11]:


df.head(10)


# ### Cleaning

# In[14]:


def clean_text(headline):
  le=WordNetLemmatizer()
  word_tokens=word_tokenize(headline)
  tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
  cleaned_text=" ".join(tokens)
  return cleaned_text


# In[15]:


# time taking
df['Abstrak_indo_cleaned']=df['Abstrak_indo'].apply(clean_text)


# In[16]:


df.head()


# In[17]:


df.drop(['Abstrak_indo'],axis=1,inplace=True)


# In[18]:


df.head()


# In[19]:


df['Abstrak_indo_cleaned'][0]


# In[20]:


vect =TfidfVectorizer(stop_words=stop_words,max_features=1000) 
# to play with. min_df,max_df,max_features etc...


# In[21]:


vect_text=vect.fit_transform(df['Abstrak_indo_cleaned'])


# In[22]:


print(vect_text.shape)
print(vect_text)


# In[23]:


idf=vect.idf_


# In[25]:


dd=dict(zip(vect.get_feature_names(), idf))
l=sorted(dd, key=(dd).get)
# print(l)
print(l[0],l[-1])
print(dd['yang'])
print(dd['wajah'])  # police is most common and forecast is least common among the news headlines.


# ### LSA

# In[26]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[27]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[28]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")

