import numpy as np
import cv2
from preprocess import load_dataset, get_cat_dog_data
from classify import DogCatClassifier
import keras.backend as K
from utilities import serialize_object

test_dir = '/home/agumbira/dev/data/dog_cat_kaggle/test/'
# dataset_path = '/home/akbar/dev/python/BOWImageClassifier/model/dog_cat_kaggle/training_data_kaze_200.dat'
# dataset_path = '/home/agumbira/dev/python/BOWImageClassifier/model/dog_cat_kaggle/training_data_kaze_200.dat'


# Prepare the data
img_rows, img_cols = 200, 1
detector = cv2.KAZE_create()
codebook_path = '/home/agumbira/dev/python/BOWImageClassifier/model/dog_cat_kaggle/codebook_kaze_200.pkl'
test_data_output_path = '/home/agumbira/dev/python/BOWImageClassifier/model/dog_cat_kaggle/test_data_kaze_200.pkl'
id, data, _ = get_cat_dog_data(detector, codebook_path, test_dir)
# Save the test data for later analysis!
serialize_object((id, data, _), test_data_output_path)

data, data1 = np.array(data), np.array(data)
if K.image_data_format() == 'channels_first':
    data1 = data1.reshape(data1.shape[0], 1, img_rows, img_cols)
else:
    data1 = data1.reshape(data1.shape[0], img_rows, img_cols, 1)

model_cnn71 = '/home/agumbira/dev/python/BOWImageClassifier/model/dog_cat_kaggle/cnn_kaze_71.h5'
model_ann72 = '/home/agumbira/dev/python/BOWImageClassifier/model/dog_cat_kaggle/ann_kaze_72.h5'
model_ann73 = '/home/agumbira/dev/python/BOWImageClassifier/model/dog_cat_kaggle/ann_kaze_73.h5'

classifier1 = DogCatClassifier(model_cnn71)
classifier2 = DogCatClassifier(model_ann72)
classifier3 = DogCatClassifier(model_ann73)
pred_label1 = classifier1.predict(data1)
pred_label2 = classifier2.predict(data)
pred_label3 = classifier2.predict(data)

id_clean = np.char.replace(id, '.jpg', '').astype(int)
pred = pred_label1 + pred_label2 + pred_label3
result = np.zeros(pred.shape[0])
result[pred == 3] = 1
result[pred == 0] = 0
result[pred == 1] = 0.2
result[pred == 2] = 0.8

print 'tes'
data = np.column_stack((id_clean, result))
data = data[data[:, 0].argsort()]
np.savetxt('submission.csv', data, delimiter=',', fmt='%i, %1.1f',
           header='id,label', comments='')

print 'test2'
