# coding=utf-8
import os
import cv2
import argparse
import sys
import time
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from preprocess import load_dataset


def train_model(alg, dataset, data_label, test_size=0):
    # Split the dataset into training and test set
    train_data, test_data, train_label, test_label = train_test_split(
        dataset, data_label, test_size=test_size)

    classifier = None
    if alg.lower() == 'svm':
        classifier = cv2.ml.SVM_create()

    classifier.train(
        np.array(train_data), cv2.ml.ROW_SAMPLE, np.array(train_label))

    if test_size != 0:
        test_predicts = classifier.predict(np.array(test_data))
        test_predicts = test_predicts[1].astype(int).flatten().tolist()
        print 'Model Accuracy: %s' % accuracy_score(test_label, test_predicts)
        print 'Confusion Matrix: '
        print confusion_matrix(test_label, test_predicts)

    return classifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Training the model and serialize it from dataset file')
    parser.add_argument(
        '-a', '--alg',
        help='Classification algorithm [svm | ann]',
        required=True)
    parser.add_argument(
        '-t', '--testsize',
        help='Split test size percentage',
        required=False)
    parser.add_argument(
        '-d', '--data', help='Dataset file', required=True)
    parser.add_argument(
        '-o', '--output', help='The output trained model file', required=True)
    args = vars(parser.parse_args())

    # Classification algorithm
    alg = args['alg'].lower()
    if alg != 'svm' and alg != 'ann':
        print 'Wrong -a args. Must be svm or ann.'
        sys.exit()

    # Test size
    test_size = 0
    if args['testsize']:
        test_size = float(args['testsize'])

    # Load dataset
    dataset_path = os.path.join(os.curdir, args['data'])
    if not os.path.exists(dataset_path):
        print 'Dataset: %s under model dir does not exist.' % dataset_path
        sys.exit()
    dataset, data_label = load_dataset(dataset_path)

    # Output trained model file
    output = os.path.join(os.curdir, args['output'])

    # Do the training
    print 'Training the model....'
    start = time.time()
    model = train_model(alg, dataset, data_label, test_size)
    model.save(output)
    print 'Elapsed time: %s sec' % (time.time() - start)

