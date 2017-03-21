# coding=utf-8
import os
import time
import sys
import pickle
import argparse
import cv2
from codebook import load_codebook


def get_bow_extractor(feature_detector, codebook):
    # Using FLANN matcher to match features
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)

    bow_extract = cv2.BOWImgDescriptorExtractor(feature_detector, flann_matcher)
    bow_extract.setVocabulary(codebook)
    return bow_extract


def preprocess_image(feature_detector, codebook_filename, output_path):
    """Represent images as histogram of visual codewords."""
    codebook = load_codebook(codebook_filename)
    bow_extractor = get_bow_extractor(feature_detector, codebook)
    # Training data
    labels = {'badminton': 1, 'bocce': 2, 'croquet': 3, 'polo': 4, 'rowing': 5,
              'RockClimbing': 6, 'sailing': 7, 'snowboarding': 8}
    training_data, training_labels = [], []
    for image_dir in labels.keys():
        path = os.path.join(os.curdir, 'images', 'training', image_dir)
        files = os.listdir(path)
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(path, file)
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                keypoints = feature_detector.detect(gray, None)
                descriptor = bow_extractor.compute(gray, keypoints)
                training_data.extend(descriptor)
                training_labels.append(labels[image_dir])

    with open(output_path, "wb") as f:
        pickle.dump((training_data, training_labels), f)


def load_dataset(dataset_path):
    with open(dataset_path, "rb") as f:
        training_data, training_label = pickle.load(f)

    return training_data, training_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocess images into dataset')
    parser.add_argument(
        '-a', '--alg', help='Descriptors algorithm', required=True)
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

    # Codebook
    codebook_path = os.path.join(os.curdir, 'model', args['cbook'])
    if not os.path.exists(codebook_path):
        print 'Codebook: %s under model dir does not exist.' % codebook_path
        sys.exit()

    # Output dataset file
    output = os.path.join(os.curdir, 'model', args['output'])

    # Do the preprocessing
    print 'Preprocessing images....'
    start = time.time()
    preprocess_image(detector, args['cbook'], output)
    print 'Elapsed time: %s sec' % (time.time() - start)

