# coding=utf-8
import os
import time
import sys
import argparse
import cv2
from codebook import load_codebook
from utilities import load_serialized_object, serialize_object


def get_bow_extractor(feature_detector, codebook):
    # Using FLANN matcher to match features
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)

    bow_extract = cv2.BOWImgDescriptorExtractor(feature_detector, flann_matcher)
    bow_extract.setVocabulary(codebook)
    return bow_extract


def preprocess_image(feature_detector, bow_extractor, image):
    """Represent an image as histogram of visual codewords"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = feature_detector.detect(gray, None)
    descriptor = bow_extractor.compute(gray, keypoints)
    return descriptor


def preprocessing_images(feature_detector, codebook_filename, image_dir):
    """Represent training images as histogram of visual codewords
        accompanied by the label."""
    codebook = load_codebook(codebook_filename)
    bow_extractor = get_bow_extractor(feature_detector, codebook)
    # Training data
    labels = {'badminton': 1, 'bocce': 2, 'croquet': 3, 'polo': 4, 'rowing': 5,
              'RockClimbing': 6, 'sailing': 7, 'snowboarding': 8}
    training_data, training_labels = [], []
    for group_dir in labels.keys():
        path = os.path.join(os.curdir, image_dir, group_dir)
        files = os.listdir(path)
        for file in files:
            if file.lower().endswith('.jpg'):
                image_path = os.path.join(path, file)
                image = cv2.imread(image_path)
                descriptor = preprocess_image(
                    feature_detector, bow_extractor, image)
                training_data.extend(descriptor)
                training_labels.append(labels[group_dir])

    return training_data, training_labels


def load_dataset(dataset_path):
    """Load the dataset from the path."""
    training_data, training_label = load_serialized_object(dataset_path)
    return training_data, training_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocess images into dataset')
    parser.add_argument(
        '-a', '--alg', help='Descriptors algorithm', required=True)
    parser.add_argument(
        '-i', '--input', help='Input images root directory', required=True)
    parser.add_argument(
        '-c', '--cbook', help='Codebook filename (under model dir)',required=True)
    parser.add_argument(
        '-o', '--output', help='The output dataset file', required=True)
    args = vars(parser.parse_args())

    detector = None
    if args['alg'].lower() == 'sift':
        detector = cv2.xfeatures2d.SIFT_create()
    elif args['alg'].lower() == 'kaze':
        detector = cv2.KAZE_create()
    else:
        print 'Wrong -a args. Must be sift or kaze.'
        sys.exit()

    # Input directory
    image_dir = os.path.join(os.curdir, args['input'])
    if not os.path.exists(image_dir) and os.path.isdir(image_dir):
        print 'Root images dir: %s does not exist.' % image_dir
        sys.exit()

    # Codebook
    codebook_path = os.path.join(os.curdir, args['cbook'])
    if not os.path.exists(codebook_path):
        print 'Codebook: %s under root dir does not exist.' % codebook_path
        sys.exit()

    # Output dataset file
    output = os.path.join(os.curdir, args['output'])

    # Do the preprocessing and serialize it
    print 'Preprocessing images....'
    start = time.time()
    dataset, data_label = preprocessing_images(detector, args['cbook'], image_dir)
    serialize_object((dataset, data_label), output)
    print 'Elapsed time: %s sec' % (time.time() - start)

