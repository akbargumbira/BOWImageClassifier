# BOWImageClassifier
Image classification using Bag of Words concept


## Building Codebook
python codebook.py -i [images_dir] -o [output_file] -a [sift|kaze] -s 
[vocab_size] 
python codebook.py -i images/training -o codebook_kaze_10.pkl -a kaze -s 10

## Preprocessing Images
python codebook.py -i  -a [sift|kaze] -c [codebook_file] -o [output_file] 
As an example:
python preprocess.py -a sift -c codebook.pkl -o dataset.pkl
