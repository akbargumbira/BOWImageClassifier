# coding=utf-8
import os
import cv2
import argparse
import sys
import time
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from nolearn.dbn import DBN
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
from keras import backend as K

import numpy as np
from preprocess import load_dataset


def train_dog_cat_kaggle(dataset, data_label, test_size=0):
    batch_size = 1024
    img_rows, img_cols = 200, 1
    input_shape = (1, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (img_rows, img_cols, 1)
    n_class = np.unique(data_label).size
    n_feature = dataset[0].shape[0]

    # Split the dataset into training and test set
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    train_index, test_index = splitter.split(dataset, data_label).next()
    train_data, test_data = np.array(dataset)[train_index], np.array(dataset)[
        test_index]
    train_label, test_label = np.array(data_label)[train_index], \
                              np.array(data_label)[test_index]

    train_label = to_categorical(train_label, n_class)
    test_label = to_categorical(test_label, n_class)

    # Using CNN
    # if K.image_data_format() == 'channels_first':
    #     train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
    #     test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
    # else:
    #     train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
    #     test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)

    # model = Sequential()
    # model.add(
    #     Conv2D(32, kernel_size=(3, 1), activation='relu',
    #            input_shape=input_shape))
    # model.add(Conv2D(64, (3, 1), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 1)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(n_class, activation='softmax'))
    # model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=keras.optimizers.Adadelta(),
    #               metrics=['accuracy'])
    # model.fit(train_data, train_label,
    #           batch_size=batch_size,
    #           epochs=50,
    #           verbose=1,
    #           validation_data=(test_data, test_label))

    model = Sequential()
    model.add(Dense(n_feature, input_dim=n_feature, activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(n_feature * 2, input_dim=n_feature, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(n_feature / 2, activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(n_feature / 4, activation='sigmoid'))
    model.add(Dense(n_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop',
                       metrics=['accuracy'])
    model.fit(train_data, train_label,
              batch_size=batch_size,
              epochs=500,
              verbose=1,
              validation_data=(test_data, test_label))

    return model


def train_model(alg, dataset, data_label, test_size=0):
    n_class = np.unique(data_label).size
    n_feature = dataset[0].shape[0]

    # Split the dataset into training and test set
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    train_index, test_index = splitter.split(dataset, data_label).next()
    train_data, test_data = np.array(dataset)[train_index], np.array(dataset)[test_index]
    train_label, test_label = np.array(data_label)[train_index], np.array(data_label)[test_index]

    classifier = None
    if alg.lower() == 'svm':
        classifier = cv2.ml.SVM_create()
        classifier.setKernel(cv2.ml.SVM_RBF)
        classifier.setType(cv2.ml.SVM_C_SVC)
        classifier.setGamma(15.383)
        classifier.setC(50)
        # Training
        classifier.train(
            train_data, cv2.ml.ROW_SAMPLE, np.array(train_label))
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
        # Training
        ann_labels = []
        for label in train_label:
            ann_label = []
            for i in range(n_class):
                if i + 1 == label:
                    ann_label.append(1)
                else:
                    ann_label.append(0)
            ann_labels.append(ann_label)

        classifier.train(
            train_data,
            cv2.ml.ROW_SAMPLE,
            np.array(ann_labels, dtype=np.float32)
        )
    elif alg.lower() == 'knn':
        classifier = cv2.ml.KNearest_create()
        # Training
        classifier.train(
            train_data, cv2.ml.ROW_SAMPLE, np.array(train_label))
    elif alg.lower() == 'dbn':
        classifier = DBN(
            [n_feature, n_feature/2, n_class],
            learn_rates=0.1,
            minibatch_size=32,
            learn_rates_pretrain=0.05,
            epochs=150,
            verbose=1,

        )
        classifier.fit(train_data, train_label.astype('int0'))
    elif alg.lower() == 'keras':
        classifier = Sequential()
        classifier.add(Dense(n_feature, input_dim=n_feature,
                             activation='relu'))
        classifier.add(Dense(n_feature, activation='relu'))
        classifier.add(Dense(n_feature/2, activation='relu'))
        classifier.add(Dense(n_feature/4, activation='relu'))
        classifier.add(Dense(n_class, activation='softmax'))
        classifier.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
        binary_labels = keras.utils.to_categorical(train_label-1,
                                                   num_classes=n_class)
        test_binary_labels = keras.utils.to_categorical(test_label-1,
                                                   num_classes=n_class)
        classifier.fit(train_data, binary_labels, epochs=100)

    # Get model accuracy on training
    get_metrics(classifier, alg, train_data, train_label, n_class)

    # Get model accuracy on test
    if test_size != 0:
        get_metrics(classifier, alg, test_data, test_label, n_class)

    return classifier


def get_metrics(model, alg, data, label, n_class):
    if alg.lower() == 'knn':
        retval, test_predicts, neigh_resp, dists = model.findNearest(
            data, 11)
        test_predicts = test_predicts.astype(int).flatten().tolist()
    elif alg.lower() == 'ann':
        ret, resp = model.predict(data)
        test_predicts = resp.argmax(-1) + 1
        test_predicts = test_predicts.astype(int).flatten().tolist()
    elif alg.lower() == 'dbn':
        test_predicts = model.predict(data)
    elif alg.lower() == 'keras':
        test_predicts = model.predict(data)
        test_predicts = test_predicts.argmax(-1) + 1
        test_predicts = test_predicts.astype(int).flatten().tolist()
    else:
        test_predicts = model.predict(data)
        test_predicts = test_predicts[1].astype(int).flatten().tolist()
    print 'Model Accuracy: %s' % accuracy_score(label, test_predicts)
    print 'Confusion Matrix: '
    print confusion_matrix(label, test_predicts)


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
    # if alg != 'svm' and alg != 'ann' and alg != 'knn' and alg != 'dbn':
    #     print 'Wrong -a args. Must be svm or ann.'
    #     sys.exit()

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

