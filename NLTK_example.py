import pandas as pd
from nltk.tokenize import TweetTokenizer
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re


data_path_training='../benchmark_system/SemEval2018-T3-train-taskA.txt'
trainingData = pd.read_csv(data_path_training, sep='\t')

data_path_test='../datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt'
testData = pd.read_csv(data_path_test, sep='\t')
trainingData.drop('Tweet index', axis=1, inplace=True)
testData.drop('Tweet index', axis=1, inplace=True)



arrayTraing = trainingData.values
X_training = arrayTraing[:, 1]
Y_training = arrayTraing[:, 0].astype(int)

arrayTest = testData.values
X_test = arrayTest[:, 1]
Y_test = arrayTest[:, 0].astype(int)

stopWords = set(stopwords.words('english'))
tokenizer = TweetTokenizer()
stemmer = SnowballStemmer("english")
X_training ="Iam getting hungry. @Never #the #less #iI hate my self hahahahaha searching"

def tweet_preprocessing(tweet):
    tweet = tweet.lower()
    mentions = re.findall(r'(@[A-Za-z0-9_]+)', tweet)
    hashtags = re.findall(r'(#[A-Za-z0-9_]+)', tweet)
    print(mentions)
    print(hashtags)
    return tweet



def feature_extraction(tweet):
    features = []
    nouncount=0
    verbcount=0
    tweetToken = [token for token in tokenizer.tokenize(tweet) if token not in stopWords]
    print(tweetToken)
    pos_tags = nltk.pos_tag(tweetToken)
    print(pos_tags)
    stemmingWords = [stemmer.stem(token) for token in tweetToken]
    print(stemmingWords)
    for token, pos in pos_tags:
        if(pos=="NN"):
            nouncount += 1
       # print(token)

    tokenCount = len(tweetToken)
    features.append([tokenCount, nouncount, verbcount])
    print(features)

X_training = tweet_preprocessing(X_training)

feature_extraction(X_training)

