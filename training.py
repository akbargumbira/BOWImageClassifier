# coding=utf-8
import os
import cv2
import argparse
import sys
import time
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from preprocess import load_dataset


def train_model(alg, dataset, data_label, test_size=0):
    n_class = np.unique(data_label).size
    n_feature = dataset[0].shape[0]

    # Split the dataset into training and test set
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size)
    train_index, test_index = splitter.split(dataset, data_label).next()
    train_data, test_data = np.array(dataset)[train_index], np.array(dataset)[test_index]
    train_label, test_label = np.array(data_label)[train_index], \
                              np.array(data_label)[test_index]
    # train_data, test_data, train_label, test_label = train_test_split(
    #     dataset, data_label, test_size=test_size)

    train_data = np.array(train_data, dtype=np.float32)
    test_data = np.array(test_data, dtype=np.float32)
    # # Scale training data
    # scaler = preprocessing.StandardScaler().fit(train_data)
    # train_data = scaler.transform(train_data)
    # test_data = scaler.transform(test_data)

    classifier = None
    if alg.lower() == 'svm':
        classifier = cv2.ml.SVM_create()
        classifier.setKernel(cv2.ml.SVM_RBF)
        classifier.setType(cv2.ml.SVM_C_SVC)
        classifier.setGamma(15.383)
        classifier.setC(50)
    elif alg.lower() == 'ann':
        classifier = cv2.ml.ANN_MLP_create()
        classifier.setLayerSizes(
            np.array([n_feature, n_feature/2, n_class], dtype=np.uint8))
        classifier.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
        classifier.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
        classifier.setBackpropWeightScale(0.01)
        classifier.setBackpropMomentumScale(0.8)
        classifier.setTermCriteria(
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 0.00001))
    elif alg.lower() == 'knn':
        classifier = cv2.ml.KNearest_create()

    # Training
    if alg.lower() == 'ann':
        ann_labels = []
        for label in train_label:
            ann_label = []
            for i in range(n_class):
                if i+1 == label:
                    ann_label.append(1)
                else:
                    ann_label.append(0)
            ann_labels.append(ann_label)

        classifier.train(
            train_data,
            cv2.ml.ROW_SAMPLE,
            np.array(ann_labels, dtype=np.float32)
        )
    elif alg.lower() == 'svm':
        classifier.train(
            train_data, cv2.ml.ROW_SAMPLE, np.array(train_label))
    else:
        classifier.train(
            train_data, cv2.ml.ROW_SAMPLE, np.array(train_label))

    # Get model accuracy
    if test_size != 0:
        if alg.lower() == 'knn':
            retval, test_predicts, neigh_resp, dists = classifier.findNearest(
                test_data, 11)
            test_predicts = test_predicts.astype(int).flatten().tolist()
        elif alg.lower() == 'ann':
            ret, resp = classifier.predict(test_data)
            test_predicts = resp.argmax(-1) + 1
            test_predicts = test_predicts.astype(int).flatten().tolist()
        else:
            test_predicts = classifier.predict(test_data)
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
    if alg != 'svm' and alg != 'ann' and alg != 'knn':
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

