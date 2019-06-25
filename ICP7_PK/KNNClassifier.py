# Importing libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords

# Reading data imported and identifying training and test data set
twenty_train = fetch_20newsgroups(subset='train', shuffle=True, categories=['alt.atheism', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'])
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)

# Creating matrix of terms present within the document(s) available
countVec = CountVectorizer()
x_train = countVec.fit_transform(twenty_train.data)
x_test = countVec.transform(twenty_test.data)

# # Identifying specific features
# print(list(twenty_train.target_names))
# target = ['alt.atheism', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']

# Building Multinomial NB Model on training data set
clfModel = MultinomialNB()
clfModel.fit(x_train, twenty_train.target)

# Predicting the model fit on test data set
clfPredcit = clfModel.predict(x_test)

# Computing the score of the model predicted to evaluate performance of the model fit
clfScore = metrics.accuracy_score(twenty_test.target, clfPredcit)
print('The score obtained for Multinomial NB model is', clfScore)

# Building K-Nearest Neighbours Model on training data set
knnModel = KNeighborsClassifier()
knnModel.fit(x_train, twenty_train.target)

# Predicting the model fit on test data set
knnPredict = knnModel.predict(x_test)

# Computing the score of the model predicted to evaluate performance of the model fit
knnScore = metrics.accuracy_score(twenty_test.target, knnPredict)
print('The score obtained for K-Nearest Neighbours model is', knnScore)

# TFIDF vectorization updated to accommodate for bigrams
bigramVec = TfidfVectorizer(ngram_range=(1, 2))
x_train_bigram = bigramVec.fit_transform(twenty_train.data)
x_test_bigram = bigramVec.transform(twenty_test.data)

# Revised models and their accuracy using bigram vectorization of data
# Multinomial NB
clfBigram = clfModel.fit(x_train_bigram, twenty_train.target)
clfPredcitBigram = clfModel.predict(x_test_bigram)
clfScoreBigram = metrics.accuracy_score(twenty_test.target, clfPredcitBigram)
print('The score obtained for Multinomial NB model when TFIDF vectorization has been updated to bigram is', clfScoreBigram)

# K-Nearest Neighbours
knnBigram = knnModel.fit(x_train_bigram, twenty_train.target)
knnPredictBigram = knnModel.predict(x_test_bigram)
knnScoreBigram = metrics.accuracy_score(twenty_test.target, knnPredictBigram)
print('The score obtained for K-Nearest Neighbours model is', knnScoreBigram)

# Stop word
# TFIDF vectorization updated to accommodate for stop word 'english'
stopwordVec = TfidfVectorizer(stop_words='english')
x_train_stopWord = stopwordVec.fit_transform(twenty_train.data)
x_test_stopWord = stopwordVec.transform(twenty_test.data)

# Revised models and their accuracy when stopword english is applied on data
# Multinomial NB
clfStopWord = clfModel.fit(x_train_stopWord, twenty_train.target)
clfPredcitStopWord = clfModel.predict(x_test_stopWord)
clfScoreStopWord = metrics.accuracy_score(twenty_test.target, clfPredcitStopWord)
print('The score obtained for Multinomial NB model when stop word english is updated is', clfScoreStopWord)

# K-Nearest Neighbours
knnStopWord = knnModel.fit(x_train_stopWord, twenty_train.target)
knnPredictStopWord = knnModel.predict(x_test_stopWord)
knnScoreStopWord = metrics.accuracy_score(twenty_test.target, knnPredictStopWord)
print('The score obtained for K-Nearest Neighbours model when stop word english is updated is', knnScoreStopWord)

# Stop word
# TFIDF vectorization updated to accommodate for stop word 'english'
stopwordsVec = TfidfVectorizer(stop_words=['a', 'The', 'is', 'with', 'has', 'an', 'did', 'were'])
x_train_stopWords = stopwordsVec.fit_transform(twenty_train.data)
x_test_stopWords = stopwordsVec.transform(twenty_test.data)

# Revised models and their accuracy when multiple propositions are used as stop words and applied to data
# Multinomial NB
clfStopWords = clfModel.fit(x_train_stopWords, twenty_train.target)
clfPredcitStopWords = clfModel.predict(x_test_stopWords)
clfScoreStopWords = metrics.accuracy_score(twenty_test.target, clfPredcitStopWords)
print('The score obtained for Multinomial NB model when stop words are updated is', clfScoreStopWords)

# K-Nearest Neighbours
knnStopWords = knnModel.fit(x_train_stopWords, twenty_train.target)
knnPredictStopWords = knnModel.predict(x_test_stopWords)
knnScoreStopWords = metrics.accuracy_score(twenty_test.target, knnPredictStopWords)
print('The score obtained for K-Nearest Neighbours model when stop words are updated is', knnScoreStopWords)