# coding=utf-8
import os
import fnmatch
import time
import sys
import argparse
import cv2
from codebook import load_codebook
from utilities import load_serialized_object, serialize_object


def get_bow_extractor(feature_detector, codebook):
    """Get the bag of words extractor object.

    :param feature_detector: The feature detector object.
    :type feature_detector: object

    :param codebook: The codebook object.
    :type codebook: object
    """
    # Using FLANN matcher to match features
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
    # Create the bow extractor
    bow_extract = cv2.BOWImgDescriptorExtractor(feature_detector, flann_matcher)
    bow_extract.setVocabulary(codebook)
    return bow_extract


def get_histogram(feature_detector, bow_extractor, image):
    """Represent an image as histogram of visual codewords.

    :param feature_detector: The feature detector object
    :type feature_detector: object

    :param bow_extractor: The BOW extractor object.
    :type bow_extractor: object

    :param image: The image instance.
    :type image: numpy.ndarray

    :return: The histogram of the image.
    :rtype:  numpy.ndarray
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = feature_detector.detect(gray, None)
    histogram = bow_extractor.compute(gray, keypoints)
    return histogram


def get_uiuc_training_data(feature_detector, codebook_path, image_dir):
    """Represent UIUC training images as histogram of visual codeword
    accompanied by the label."""
    codebook = load_codebook(codebook_path)
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
                histogram = get_histogram(
                    feature_detector, bow_extractor, image)
                training_data.extend(histogram)
                training_labels.append(labels[group_dir])

    return training_data, training_labels


def get_cat_dog_data(feature_detector, codebook_path, image_dir):
    """Represent cat vs dog kaggle training images as histogram of visual
    codeword accompanied by the label."""
    codebook = load_codebook(codebook_path)
    bow_extractor = get_bow_extractor(feature_detector, codebook)
    # Training data
    id, training_data, training_labels = [], [], []
    for root, dirnames, filenames in os.walk(image_dir):
        filenames = fnmatch.filter(filenames, '*.[Jj][Pp][Gg]')
        for index, filename in enumerate(filenames):
            image_path = os.path.join(root, filename)
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                histogram = get_histogram(
                    feature_detector, bow_extractor, image)
                if histogram is not None:
                    id.append(filename)
                    training_data.extend(histogram)
                    if 'dog' in filename.lower():
                        training_labels.append(1)
                    else:
                        training_labels.append(0)

    return id, training_data, training_labels


def load_dataset(dataset_path):
    """Load the dataset from the path."""
    training_data, training_label = load_serialized_object(dataset_path)
    return training_data, training_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocess images into dataset')
    parser.add_argument(
        '-p', '--problem', help='Problem - uiuc or kaggle', required=True)
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
    codebook_path = args['cbook']
    if not os.path.exists(codebook_path):
        print 'Codebook: %s does not exist.' % codebook_path
        sys.exit()

    # Output dataset file
    output = args['output']

    # Do the preprocessing and serialize it
    print 'Preprocessing images....'
    start = time.time()
    if args['problem'].lower() == 'uiuc':
        dataset, data_label = get_uiuc_training_data(
            detector, args['cbook'], image_dir)
    elif args['problem'].lower() == 'kaggle':
        id, dataset, data_label = get_cat_dog_data(
            detector, args['cbook'], image_dir)
    serialize_object((dataset, data_label), output)
    print 'Elapsed time: %s sec' % (time.time() - start)

