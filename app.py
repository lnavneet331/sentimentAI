import streamlit as st
import pickle
import sklearn
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cv = CountVectorizer(max_features=5000)
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
clf=MultinomialNB()
from time import sleep
data=pickle.load(open("dataframe.pkl", "rb"))
features_dict = pickle.load(open("features_dict.pkl", "rb"))

def vectorize(dataframe):
    X = cv.fit_transform(dataframe.review).toarray()
    data.sentiment=le.fit_transform(dataframe.sentiment)
    y=data.iloc[:,-1].values
    return X, y

def stemming(text, stemmer=PorterStemmer()):
    stem_word=[]
    for i in text.split():
        stem_word.append(stemmer.stem(i))
    return stem_word

#store the text in BoW(bag of words)
def vectorBuild(val):
    a=np.zeros(5000)
    for i in range(len(val)):
        if val[i] in features_dict:
            a[features_dict[val[i]]] += 1
    a = a.reshape(1, -1)
    return a

def modelFunction(X, y, a):
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=data.sentiment)
    clf.fit(train_X, train_y)
    pred = clf.predict(a)
    return pred

st.title("Sentiment Analysis Prediction")

container = st.container()
container.write("You need to press enter everytime, empty textbox will show \'Positve Sentiment\'")
container.write()
selected_text = container.text_input("Enter the text that you want to test")
container.write("Processing...")

a = stemming(selected_text)
a = vectorBuild(a)
X, y = vectorize(data)
model = modelFunction(X, y, a)

if model[0] == 1:
    container.write("Positive Sentiment")
elif model[0] == 0:
    container.write("Negative Sentiment")