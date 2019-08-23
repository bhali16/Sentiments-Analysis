import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

#Reading Data and naming Columns
df = pd.read_csv("scm.txt", sep='\t', names=['liked', 'comment'])

#Removing Stops Words
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)

y=df.liked

X = vectorizer.fit_transform(df['comment'].values.astype('U'))

#Split Data in to Test Train
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 42)

#Using Classifier on Data
clf = naive_bayes.MultinomialNB()
clf.fit(X_train, y_train)
roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])

#Testing Model with single input
movies_review_array = np.array(["the quality of headphones is bad"])
movies_review_vector = vectorizer.transform(movies_review_array)
print(clf.predict(movies_review_vector))

#Applying Model on Test Data
df2 = pd.read_csv("testdata.txt", sep='\n', names=['comment'])
movies_review_array = np.array(df2.comment)
movies_review_vector = vectorizer.transform(movies_review_array)
prediction = clf.predict(movies_review_vector)
print(prediction)