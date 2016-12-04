from __future__ import print_function
import nltk
import random
import string
import numpy as np
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.model_selection import cross_val_score
from sklearn import svm, preprocessing
from nltk import word_tokenize,sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import movie_reviews
from nltk.tokenize import RegexpTokenizer

from nmf_kl import KLdivNMF
from rlx_nmf_kl import RlxKLdivNMF



random.seed(0)
token_dict = {}

i=0
for category in movie_reviews.categories():
	for fileid in movie_reviews.fileids(category):
		token_dict[i] = movie_reviews.raw(fileid)
		i = i+1

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(text)
	#stems = stem_tokens(tokens, stemmer)
	return tokens

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

#print token_dict.values()

n_features = 5000
n_topics = 80
n_top_words = 20
max_iter = 300


countvec = CountVectorizer(tokenizer=tokenize)
raw_tdmatrix = countvec.fit_transform(token_dict.values())
raw_vocab = countvec.vocabulary_
feature_names = countvec.get_feature_names()

sorted_idx = raw_tdmatrix.sum(0).argsort().tolist()[0]
vocab_idx = sorted_idx[-5050:-50]

vocab = []
for idx in vocab_idx:
	vocab.append(feature_names[idx])

countvec = CountVectorizer(tokenizer=tokenize, vocabulary=vocab)
tdmatrix = countvec.fit_transform(token_dict.values())
feature_names = countvec.get_feature_names()

# Fit the Relaxed NMF model
print("Fitting the Relaxed NMF model")
t0 = time()
rlx_nmf = RlxKLdivNMF(n_components=n_topics, random_state=1, max_iter=max_iter, init='random', rho=500.0)
rlx_nmf.fit(tdmatrix)
print("done in %0.3fs." % (time() - t0))
#print("\nTopics in L2-NMF model:")
#print_top_words(rlx_nmf, feature_names, n_top_words)


# Fit the L2-NMF model
print("Fitting the L2-NMF model")
t0 = time()
nmf = NMF(n_components=n_topics, random_state=1, max_iter=max_iter, init='nndsvd', solver='cd')
nmf.fit(tdmatrix)
print("done in %0.3fs." % (time() - t0))
#print("\nTopics in L2-NMF model:")
#print_top_words(nmf, feature_names, n_top_words)

# Fit the NMF model
print("Fitting the KL-NMF model")
t0 = time()
kl_nmf = KLdivNMF(n_components=n_topics, random_state=1, max_iter=max_iter, init='nndsvd')
kl_nmf.fit(tdmatrix)
print("done in %0.3fs." % (time() - t0))
#print("\nTopics in KL-NMF model:")
#print_top_words(nmf, feature_names, n_top_words)

# Fit the LDA model
t0 = time()
print("Fitting LDA models")
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=max_iter,
                                learning_method='batch', n_jobs=1,
                                evaluate_every=5, random_state=0)
lda.fit(tdmatrix)
print("done in %0.3fs." % (time() - t0))

#print("\nTopics in LDA model:")
#print_top_words(lda, feature_names, n_top_words)


# extract features
permute = random.sample(range(2000), 2000)
labels = np.ones(2000)
labels[1000:] = -1
raw_rlx_nmf_features = rlx_nmf.transform(tdmatrix)
raw_nmf_features = nmf.transform(tdmatrix)
raw_kl_nmf_features = kl_nmf.transform(tdmatrix)
raw_lda_features = lda.transform(tdmatrix)

# scale the raw features
min_max_scaler = preprocessing.MinMaxScaler()

labels = labels[permute];
rlx_nmf_features = min_max_scaler.fit_transform(raw_rlx_nmf_features[permute])
nmf_features = min_max_scaler.fit_transform(raw_nmf_features[permute])
kl_nmf_features = min_max_scaler.fit_transform(raw_kl_nmf_features[permute])
lda_features = raw_lda_features[permute]

# train svms on scaled features
print("10-fold cross-validation acc of Relaxed NMF:")
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, rlx_nmf_features, labels, cv=10)
print(np.mean(scores))

print("10-fold cross-validation acc of L2-NMF:")
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, nmf_features, labels, cv=10)
print(np.mean(scores))

print("10-fold cross-validation acc of KL-NMF:")
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, kl_nmf_features, labels, cv=10)
print(np.mean(scores))

print("10-fold cross-validation acc of LDA:")
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, lda_features, labels, cv=10)
print(np.mean(scores))

