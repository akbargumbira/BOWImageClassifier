# coding=utf-8
import os
import time
import sys
import numpy as np
import cv2
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix
from preprocess import load_dataset
from keras.models import load_model


class DogCatClassifier(object):
    """Class DigitNotDigit."""
    def __init__(self, model_path):
        """The constructor."""
        self._model = load_model(model_path)

    def predict(self, data):
        predictions = self._model.predict(data)
        label = np.argmax(predictions, 1)
        return label


def classify(classifier, test_data):
    """Classify"""
    test_predicts = classifier.predict(np.array(test_data))
    test_predicts = test_predicts[1].astype(int).flatten().tolist()
    return test_predicts

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Classify images')
    parser.add_argument(
        '-a', '--alg',
        help='Classification algorithm [svm | ann]',
        required=True)
    parser.add_argument(
        '-m', '--model', help='Model filename (under model dir)', required=True)
    parser.add_argument(
        '-d', '--data', help='Dataset test file', required=True)
    args = vars(parser.parse_args())

    # Load model
    model_path = os.path.join(os.curdir, args['model'])
    if not os.path.exists(model_path):
        print 'Classifier model: %s under model dir does not exist.' % \
              model_path
        sys.exit()

    # Classification algorithm
    alg = args['alg'].lower()
    classifier = None
    if alg == 'svm':
        classifier = cv2.ml.SVM_load(model_path)
    elif alg == 'ann':
        print 'Not yet implemented'
    else:
        print 'Wrong -a args. Must be svm or ann.'
        sys.exit()

    # Load dataset
    dataset_path = os.path.join(os.curdir, args['data'])
    if not os.path.exists(dataset_path):
        print 'Dataset: %s under root dir does not exist.' % dataset_path
        sys.exit()
    test_data, test_label = load_dataset(dataset_path)

    # Do the classification
    print 'Classifying the test data....'
    start = time.time()
    test_result = classify(classifier, test_data)
    print 'Accuracy: %s' % accuracy_score(test_label, test_result)
    print 'Confusion Matrix: '
    print confusion_matrix(test_label, test_result)

    print 'Elapsed time: %s sec' % (time.time() - start)

