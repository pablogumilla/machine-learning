#! /usr/bin/env python
import os
import numpy
import pickle

from collections import (
    Counter,
    OrderedDict,
)

from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATASET_PATH = "{base_path}/datasets/emails/train/".format(
    base_path=BASE_PATH
)
TEST_DATASET_PATH = "{base_path}/datasets/emails/test/".format(
    base_path=BASE_PATH
)
INSTANCE_LENGTH = 3000
TEST_VECTOR_LENGTH = 260
TRAIN_VECTOR_LENGTH = 702


def _create_instance(dataset_path):
    email_files = os.listdir(dataset_path)
    word_list = []
    for mail in email_files:
        mail_path = "{path}{file}".format(path=dataset_path, file=mail)
        with open(mail_path, 'rb') as fd:
            lines = fd.readlines()
            if lines:
                # Mail body is always last line
                word_list += lines[-1].split()

    words_counter = Counter(word_list)
    for item in words_counter.keys():
        if (not item.isalpha() or len(item) == 1):
            del words_counter[item]

    return OrderedDict(words_counter.most_common(INSTANCE_LENGTH))


def _extract_features(dataset_path, words_counter):
    words_keys = words_counter.keys()
    email_files = os.listdir(dataset_path)
    features_matrix = numpy.zeros((len(email_files), INSTANCE_LENGTH))

    for doc_index, mail in enumerate(email_files):
        mail_path = "{path}{file}".format(path=dataset_path, file=mail)
        with open(mail_path, 'rb') as fd:
            lines = fd.readlines()
            word_list = lines[-1].split()

            for word in word_list:
                word_index = words_keys.index(word) if word in words_keys else None
                if word_index:
                    features_matrix[doc_index, word_index] = word_list.count(word)

    return features_matrix

def evaluate_results(labels, results):
    evaluation = confusion_matrix(labels, results)
    print "Ham email (right, wrong): {ham}".format(
        ham=' '.join(map(str, evaluation[0]))
    )
    print "Spam email (right, wrong): {spam}".format(
        spam=' '.join(map(str, evaluation[1]))
    )

# Create a dictionary of words with its frequency
words_counter = _create_instance(TRAIN_DATASET_PATH)

# Create matrix with training instances
train_matrix = _extract_features(TRAIN_DATASET_PATH, words_counter)

# Prepare feature vectors per training mail and its labels
train_labels = numpy.zeros(TRAIN_VECTOR_LENGTH)
train_labels[TRAIN_VECTOR_LENGTH/2:TRAIN_VECTOR_LENGTH] = 1

# Multinomial Naive Bayes
mnb_classifier = MultinomialNB()
mnb_classifier.fit(train_matrix, train_labels)

# Create matrix with test instances
test_matrix = _extract_features(TEST_DATASET_PATH, words_counter)

# Prepare feature vectors per testing mail and its labels
test_labels = numpy.zeros(TEST_VECTOR_LENGTH)
test_labels[TEST_VECTOR_LENGTH/2:TEST_VECTOR_LENGTH] = 1

mnb_result = mnb_classifier.predict(test_matrix)

evaluate_results(test_labels, mnb_result)
