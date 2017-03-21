import os
import numpy as np
import cv2
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from codebook import load_codebook
from training import get_training_data, get_bow_extractor

sift_detector = cv2.xfeatures2d.SIFT_create()
# Getting Bag of Words
codebook_path = os.path.join(os.curdir, 'codebook.pkl')
vocabulary = load_codebook(codebook_path)
# Get the bow extractor
bow_extract = get_bow_extractor(sift_detector, vocabulary)

# Training
dataset, data_label = get_training_data(sift_detector, bow_extract)
train_data, test_data, train_label, test_label = train_test_split(
    dataset, data_label, test_size=0)
svm = cv2.ml.SVM_create()
svm.train(np.array(train_data), cv2.ml.ROW_SAMPLE, np.array(train_label))
# results = svm.predict(np.array(test_data))
# print classification_report(test_label, results[1].astype(int).flatten().tolist())

# Predict test
test_path = os.path.join(os.curdir, 'images/test')
test_files = os.listdir(test_path)
for test_file in test_files:
    if test_file.endswith('.jpg'):
        image_path = os.path.join(test_path, test_file)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints = sift_detector.detect(gray, None)
        des = bow_extract.compute(gray, keypoints)
        print 'Class for %s: %s' % (test_file, svm.predict(des))
