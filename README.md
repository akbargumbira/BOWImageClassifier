0# BOWImageClassifier
Image classification using Bag of Words concept


## Building Codebook
python codebook.py -i [images_dir] -o [output_file] -a [sift|kaze] -s [vocab_size]

python codebook.py -i images/training -o model/codebook_kaze_10.pkl -a kaze -s 10

## Preprocessing Images
python preprocess.py  -a [sift|kaze] -i [root_image_dir] -c [codebook_file] -o [output_file]

python preprocess.py -a sift -i images/training -c model/codebook.pkl -o model/dataset.dat


## Training
python training.py -a [svm|ann] -d [serialized_dataset_file] -o [trained_model_file] -t [test_size_percentage]

python training.py -a svm -d model/dataset.dat -o model/model_svm.xml -t 0.1

## Classification
python classify.py -a [svm|ann] -m [model_file] -d [serialized_dataset_file]

python classify.py -a svm -m model/model_svm.xml -d model/test_data.dat

