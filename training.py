# coding=utf-8
import os
import cv2
from codebook import load_codebook
from sklearn.cross_validation import train_test_split
import numpy as np


def train(feature_detector, codebook_path):
    vocabulary = load_codebook(codebook_path)
    bow_extractor = get_bow_extractor(feature_detector, vocabulary)
    dataset, data_label = get_training_data(feature_detector, bow_extractor)
    svm = cv2.ml.SVM_create()
    train_data, test_data, train_label, test_label = train_test_split(
        dataset, data_label, test_size=0.3)
    svm.train(np.array(train_data), cv2.ml.ROW_SAMPLE, np.array(train_label))

if __name__ == '__main__':
    # Training
    print 'tes'

