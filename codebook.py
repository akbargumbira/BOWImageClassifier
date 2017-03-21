# coding=utf-8
import os
import sys
import time
import fnmatch
import pickle
import argparse
import cv2


def build_codebook(input_dir, output_path, alg='sift', vocab_size=240):
    if alg.lower() == 'sift':
        detector = cv2.xfeatures2d.SIFT_create()
    elif alg.lower() == 'kaze':
        detector = cv2.KAZE_create()
    else:
        print 'Unknown algorithm. Option: sift | kaze'
        return

    bow = cv2.BOWKMeansTrainer(vocab_size)
    # Read images
    path = os.path.join(os.curdir, input_dir)
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.jpg'):
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = detector.detectAndCompute(gray, None)
            bow.add(descriptors)

    codewords = bow.cluster()
    codebook_file = open(output_path, 'wb')
    pickle.dump(codewords, codebook_file)


def load_codebook(filename):
    codebook_path = os.path.join(os.curdir, 'model', filename)
    codebook_file = open(codebook_path, 'rb')
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
    args = vars(parser.parse_args())

    # Input directory
    input_dir = os.path.join(os.curdir, args['input'])
    if not os.path.exists(input_dir):
        print 'Input directory: %s does not exist.' % input_dir
        sys.exit()

    # Output path
    output = os.path.join(os.curdir, 'model', args['output'])

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
    build_codebook(input_dir, output, alg, vocab_size)
    print 'Elapsed time: %s sec' % (time.time() - start)
