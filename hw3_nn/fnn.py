# fnn.py

import tarfile
import numpy as np
import pandas as pd
import torch
import csv
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm
import time

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
stop = stopwords.words('english')


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub(r'[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

def tokenizer(text):
    return text.split()

porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

# 1. transform imdb data into tf-idf form.
# read in data

path = "movie_data.csv"
print("Reading CSV...")
data = pd.read_csv(path)
print("Done!")
print()


# clean data of HTML formatting (taken from pg. 255) confirmed working. 
print("Beginning preprocessing:")
data['review'] = data['review'].apply(preprocessor)
print("Done!")
print()

########################################

# training logistic regression
X_train = data.loc[:25000, 'review'].values
y_train = data.loc[:25000, 'sentiment'].values
X_test = data.loc[25000:, 'review'].values
y_test = data.loc[25000:, 'sentiment'].values

tfidf = TfidfVectorizer(strip_accents=None, lowercase = False, preprocessor= None)

"""
param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]


"""
small_param_grid = [{'vect__ngram_range': [(1, 1)],
                     'vect__stop_words': [None],
                     'vect__tokenizer': [tokenizer, tokenizer_porter],
                     'clf__penalty': ['l2'],
                     'clf__C': [1.0, 10.0]},
                    {'vect__ngram_range': [(1, 1)],
                     'vect__stop_words': [stop, None],
                     'vect__tokenizer': [tokenizer],
                     'vect__use_idf':[False],
                     'vect__norm':[None],
                     'clf__penalty': ['l2'],
                  'clf__C': [1.0, 10.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(solver='liblinear'))])


gs_lr_tfidf = GridSearchCV(lr_tfidf, small_param_grid,
                        scoring='accuracy',
                        cv=5,
                        verbose=1,
                        n_jobs=-1)

start_time = time.time()
gs_lr_tfidf.fit(X_train, y_train)
end_time = time.time()

best_model = gs_lr_tfidf.best_estimator_ 
train_accuracy = best_model.score(X_train, y_train) 
test_accuracy = best_model.score(X_test, y_test) 
time_cost = end_time - start_time

print("\nBest Parameters:") 
print(gs_lr_tfidf.best_params_) 
print("\nTraining Accuracy:", train_accuracy) 
print("Test Accuracy:", test_accuracy) 
print("\nTime Cost (seconds):\n", time_cost)

print("Book output")
print(f'Best parameter set: {gs_lr_tfidf.best_params_}')
print(f'CV Accuracy: {gs_lr_tfidf.best_score_:.3f}')
clf = gs_lr_tfidf.best_estimator_
print(f'Test Accuracy: {clf.score(X_test, y_test):.3f}')