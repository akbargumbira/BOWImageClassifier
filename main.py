import os
import numpy as np
import cv2
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from codebook import load_codebook, build_codebook
from training import train_model, train_dog_cat_kaggle
from preprocess import load_dataset, get_cat_dog_data
from classify import DogCatClassifier
import keras.backend as K

#
# sift_detector = cv2.xfeatures2d.SIFT_create()
# # Getting Bag of Words
# codebook_path = os.path.join(os.curdir, 'codebook.pkl')
# vocabulary = load_codebook(codebook_path)
# # Get the bow extractor
# bow_extract = get_bow_extractor(sift_detector, vocabulary)
#
# # Training
# dataset, data_label = get_training_data(sift_detector, bow_extract)
# train_data, test_data, train_label, test_label = train_test_split(
#     dataset, data_label, test_size=0)
# svm = cv2.ml.SVM_create()
# svm.train(np.array(train_data), cv2.ml.ROW_SAMPLE, np.array(train_label))
# # results = svm.predict(np.array(test_data))
# # print classification_report(test_label, results[1].astype(int).flatten().tolist())
#
# # Predict test
# test_path = os.path.join(os.curdir, 'images/test')
# test_files = os.listdir(test_path)
# for test_file in test_files:
#     if test_file.endswith('.jpg'):
#         image_path = os.path.join(test_path, test_file)
#         image = cv2.imread(image_path)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         keypoints = sift_detector.detect(gray, None)
#         des = bow_extract.compute(gray, keypoints)
#         print 'Class for %s: %s' % (test_file, svm.predict(des))

# dataset_path = os.path.join(os.curdir, 'model/training_data_sift_240.dat')
# dataset, data_label = load_dataset(dataset_path)
# model = train_model("dbn", dataset, data_label, 0.1)
# print 'tes'

# DOG vs CAT - KAGGLE
# 1. Building the codebook from all the training images using kaze with 200
#   clusters
# build_codebook(
#     '/home/agumbira/dev/data/dog_cat_kaggle/train',
#     '/home/agumbira/dev/python/BOWImageClassifier/model/dog_cat_kaggle',
#     'kaze',
#     200,
#     verbose=True)

# 2 Preprocessing all the images into dataset using defined codebook
detector = cv2.KAZE_create()
# codebook_path = '/home/akbar/dev/python/BOWImageClassifier/model/dog_cat_kaggle/codebook_kaze_test_200.pkl'
# codebook_path = '/home/agumbira/dev/python/BOWImageClassifier/model/dog_cat_kaggle/codebook_kaze_200.pkl'
# test_dir = '/home/akbar/dev/data/dog_cat_kaggle/test/'
# test_dir1 = '/home/akbar/dev/data/dog_cat_kaggle/test_1/'
test_dir1 = '/home/agumbira/dev/data/dog_cat_kaggle/test_1/'
test_dir2 = '/home/akbar/dev/data/dog_cat_kaggle/test_2/'

# dataset_path = '/home/akbar/dev/python/BOWImageClassifier/model/dog_cat_kaggle/training_data_kaze_test_200.dat'
dataset_path = '/home/agumbira/dev/python/BOWImageClassifier/model/dog_cat_kaggle/training_data_kaze_200.dat'
# model_output = '/home/akbar/dev/python/BOWImageClassifier/model/dog_cat_kaggle/cnn_kaze.h5'
model_output = '/home/agumbira/dev/python/BOWImageClassifier/model/dog_cat_kaggle/ann_kaze.h5'
dataset, data_label = load_dataset(dataset_path)
model = train_dog_cat_kaggle(dataset, data_label, 0.1)
model.save(model_output)
# img_rows, img_cols = 20, 10
# id, data, label = get_cat_dog_data(detector, codebook_path, test_dir)
# id1, data1, label1 = get_cat_dog_data(detector, codebook_path, test_dir1)
# id2, data2, label2 = get_cat_dog_data(detector, codebook_path, test_dir2)
# data, data1, data2 = np.array(data), np.array(data1), np.array(data2)
# if K.image_data_format() == 'channels_first':
#     data = data.reshape(data.shape[0], 1, img_rows, img_cols)
#     data1 = data1.reshape(data1.shape[0], 1, img_rows, img_cols)
#     data2 = data2.reshape(data2.shape[0], 1, img_rows, img_cols)
# else:
#     data = data.reshape(data.shape[0], img_rows, img_cols, 1)
#     data1 = data2.reshape(data2.shape[0], img_rows, img_cols, 1)
#     data2 = data2.reshape(data2.shape[0], img_rows, img_cols, 1)
#
# classifier = DogCatClassifier()
# pred_label = classifier.predict(data)
# pred_label2 = classifier.predict(data2)

print 'tes'

