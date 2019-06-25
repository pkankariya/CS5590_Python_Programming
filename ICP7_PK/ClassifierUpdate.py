# Importing libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Reading data imported and identifying training and test data set
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
# print('Training Data:', twenty_train)
# print('Test Data:', twenty_test)

# Creating matrix of terms present within the document(s) available

countVec = CountVectorizer()
x_train = countVec.fit_transform(twenty_train.data)
x_test = countVec.transform(twenty_test.data)
#X_train_counts = countVec.fit_transform(twenty_train.data)

# Building Multinomial NB Model on training data set
clfModel = MultinomialNB()
clfModel.fit(x_train, twenty_train.target)
#print(clfModel)

# Predicting the model fit on test data set
clfPredcit = clfModel.predict(x_test)

# Computing the score of the model predicted to evaluate performance of the model fit
clfScore = metrics.accuracy_score(twenty_test.target, clfPredcit)
print('The score obtained for Multinomial NB model is', clfScore)

# Building SVM (Support Vector machine) Model on training data set
svmModel = SVC()
svmModel.fit(x_train, twenty_train.target)
#print(clfModel)

# Predicting the model fit on test data set
svmPredict = svmModel.predict(x_test)

# Computing the score of the model predicted to evaluate performance of the model fit
svmScore = metrics.accuracy_score(twenty_test.target, svmPredict)
print('The score obtained for SVM model is', svmScore)

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

# Support Vector Machine
svmBigram = svmModel.fit(x_train_bigram, twenty_train.target)
svmPredictBigram = svmModel.predict(x_test_bigram)
svmScoreBigram = metrics.accuracy_score(twenty_test.target, svmPredictBigram)
print('The score obtained for SVM when TFIDF vectorization has been updated to bigram is', svmScoreBigram)