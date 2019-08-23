from textblob import TextBlob
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from textblob.sentiments import NaiveBayesAnalyzer
import nltk
from textblob.classifiers import NaiveBayesClassifier


df = pd.read_csv("train.csv", names=['rating','comment', 'review'])

score = []
reviews = df.review
for a in reviews:
    blob = TextBlob(a)
    score.append(blob.sentiment)

polarity_list = []
subjectivity_list = []

for a in reviews:
    blob = TextBlob(a)
    polarity_list.append(blob.sentiment.polarity)

for a in reviews:
    blob = TextBlob(a)
    subjectivity_list.append(blob.sentiment.subjectivity)

print(subjectivity_list)
print(polarity_list)

#Plot Graph of Polarity and Subjectivity of Sentiments
x = subjectivity_list
y = polarity_list
plt.plot(x)
plt.plot(y)

#Get Keys Positive Negative and Neutral based on Polarity
p_r = 0
neg_r = 0
neu_r = 0

reviews = []
for items in polarity_list:
    if(items > 0):
        print('postive')
        reviews.append('positive')
        p_r += 1
    elif(items == 0):
        print('neutral')
        reviews.append('neutral')
        neg_r += 1
    elif(items < 0):
        print('negative')
        reviews.append('negative')
        neu_r += 1

#Plot the Positive Neutral and Negative on Pie Chart
a = p_r
b = neg_r
c = neu_r
labels = 'Postive','Neutral','Negtive'
sizes = [a,b,c]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

#Function for Min Max Normalizations
minval = min(df.rating)
maxval = max(df.rating)
ratingx=[]

def minmax(val):
    finalx = (x - minval)/(maxval-minval)
    print(finalx)

# # For Applying Naive Baye Analzer on Text Blob
# text          = "I feel the product is so good" 
# sent          = TextBlob(text)
# # The polarity score is a float within the range [-1.0, 1.0]
# # where negative value indicates negative text and positive
# # value indicates that the given text is positive.
# polarity      = sent.sentiment.polarity
# # The subjectivity is a float within the range [0.0, 1.0] where
# # 0.0 is very objective and 1.0 is very subjective.
# subjectivity  = sent.sentiment.subjectivity

# sent          = TextBlob(text, analyzer = NaiveBayesAnalyzer())
# classification= sent.sentiment.classification
# positive      = sent.sentiment.p_pos
# negative      = sent.sentiment.p_neg

# print(polarity,subjectivity,classification,positive,negative)

# #cl = NaiveBayesClassifier(traind)

pos_count = 0
pos_correct = 0
neg_count = 0
neg_correct = 0

#positive accuracy
for rv in df.review:
    analysis = TextBlob(rv)
    if analysis.sentiment.polarity >= 0.5:
        if analysis.sentiment.polarity > 0:
            pos_correct += 1
        pos_count +=1
#Negative accuracy
for rv in df.review:
    analysis = TextBlob(rv)
    if analysis.sentiment.polarity <= -0.5:
        if analysis.sentiment.polarity <= 0:
            neg_correct += 1
        neg_count +=1
        
print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100.0, pos_count))
print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100.0, neg_count))

#Positive Correct
pos_correct = 0
for pol in polarity_list:
    if pol > 0.2:
        pos_correct += 1
print("Postive Correct : ", pos_correct)

#Negative Correct
neg_correct = 0
for pol in polarity_list:
    if pol <= 0.2:
        neg_correct += 1
print("Negative Correct : ", neg_correct)