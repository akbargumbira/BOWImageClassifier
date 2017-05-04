# coding=utf-8
import os
import sys
import time
import fnmatch
import pickle
import argparse
import cv2
import numpy as np
from sklearn.cluster import KMeans


def build_codebook(
        input_dir, output_path, alg='sift', vocab_size=240, verbose=False):
    """Build the codebook (dictionary) for all the images in input dir.

    :param input_dir: The input directory containing all the images.
    :type input_dir: str

    :param output_path: The codebook output path.
    :type output_path: str

    :param alg: The feature detection & description algorithm (SIFT/KAZE).
    :type alg: str

    :param vocab_size: The vocabulary size (the number of clusters).
    :type vocab_size: int

    :param verbose: Show the status every 1% of total images.
    :type verbose: bool
    """
    if alg.lower() == 'sift':
        detector = cv2.xfeatures2d.SIFT_create()
    elif alg.lower() == 'kaze':
        detector = cv2.KAZE_create()
    else:
        print 'Unknown algorithm. Option: sift | kaze'
        return

    bow = cv2.BOWKMeansTrainer(vocab_size)
    # Read images
    for root, dirnames, filenames in os.walk(input_dir):
        if verbose:
            print 'Extracting descriptors of images in: %s ...' % root
        n_images = len(filenames)
        filenames = fnmatch.filter(filenames, '*.[Jj][Pp][Gg]')
        for index, filename in enumerate(filenames):
            n_chunk = int(round(float(n_images) / 100))
            n_chunk = 1 if n_chunk == 0 else n_chunk
            if index % n_chunk == 0 and verbose:
                print 'Processed: %s %% of images' % (index*100/n_images)
            # Get the descriptors
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = detector.detectAndCompute(gray, None)
            bow.add(descriptors)

    # Cluster all the descriptors and save it into output file
    if verbose:
        print 'Clustering all the descriptors...'
    codewords = bow.cluster()
    codebook_file = open(output_path, 'wb')
    pickle.dump(codewords, codebook_file)
    return codewords


def build_codebook_scikit(
        input_dir, output_path, alg='sift', vocab_size=240, verbose=False):
    if alg.lower() == 'sift':
        detector = cv2.xfeatures2d.SIFT_create()
    elif alg.lower() == 'kaze':
        detector = cv2.KAZE_create()
    else:
        print 'Unknown algorithm. Option: sift | kaze'
        return

    images_descriptors = []
    # Read images
    for root, dirnames, filenames in os.walk(input_dir):
        if verbose:
            print 'Extracting descriptors of images in: %s ...' % root
        n_images = len(filenames)
        filenames = fnmatch.filter(filenames, '*.[Jj][Pp][Gg]')
        for index, filename in enumerate(filenames):
            n_chunk = int(round(float(n_images) / 100))
            n_chunk = 1 if n_chunk == 0 else n_chunk
            if index % n_chunk == 0 and verbose:
                print 'Processed: %s %% of images' % (index*100/n_images)
            # Get the descriptors
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = detector.detectAndCompute(gray, None)
            images_descriptors.extend(descriptors)

    # Cluster all the descriptors and save it into output file
    if verbose:
        print 'Clustering all the descriptors...'

    kmeans = KMeans(n_clusters=vocab_size)
    kmeans.fit(images_descriptors)
    labels = kmeans.labels_
    codewords = kmeans.cluster_centers_
    # term_crit = 0, 0, 0
    # attempts = 3
    # flags = cv2.KMEANS_PP_CENTERS
    # ret, labels, centroids = cv2.kmeans(
    #     np.array(images_descriptors), vocab_size, None, term_crit, attempts, flags)
    return labels, codewords

def load_codebook(file_path):
    """Load the codebook from a file.

    :param file_path: The codebook path.
    :type: str

    :return: The codebook object.
    :rtype: object
    """
    with open(file_path, 'rb') as codebook_file:
        codewords = pickle.load(codebook_file)
    return codewords


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Building the codebook')
    parser.add_argument(
        '-i', '--input', help='The input directory', required=True)
    parser.add_argument(
        '-o', '--output', help='The output file', required=True)
    parser.add_argument(
        '-a', '--alg', help='Descriptors algorithm', required=True)
    parser.add_argument(
        '-s', '--size', help='Codebook size (default=240)', required=False)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)
    args = vars(parser.parse_args())

    # Input directory
    input_dir = args['input']
    if not os.path.exists(input_dir):
        print 'Input directory: %s does not exist.' % input_dir
        sys.exit()

    # Output path
    output = args['output']

    # Algorithm
    alg = None
    if args['alg'].lower() == 'sift':
        alg = 'sift'
    elif args['alg'].lower() == 'kaze':
        alg = 'kaze'
    else:
        print 'Wrong -a args. Must be sift or kaze.'
        sys.exit()

    # Vocab size
    vocab_size = None
    if args['size']:
        vocab_size = int(args['size'])

    print 'Building the codebook...'
    start = time.time()
    build_codebook(input_dir, output, alg, vocab_size, args['verbose'])
    print 'Elapsed time: %s sec' % (time.time() - start)
