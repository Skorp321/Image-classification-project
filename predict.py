import argparse
import json
import numpy as np
import tensorflow as tf
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('image', help='path to image')
ap.add_argument('model', help='my DL model')
ap.add_argument('jes', help='path to jesun file')
ap.add_argument('k_top', default=1, type=int, help='top N values')
args = vars(ap.parse_args())

reloaded_keras_model = tf.keras.models.load_model(args['model'])

with open(args['jes'], 'r') as f:
    class_names = json.load(f)

def process_image(test_image):
    test_image = tf.cast(test_image, tf.float32)
    test_image = tf.image.resize(test_image, [224, 224])/ 255
    return test_image.numpy()

def predict(processed_test_image, model, k):
    batcht_im = processed_test_image[np.newaxis]
    preds = model.predict(batcht_im)
    preditions = tf.math.top_k(preds, k)
    return preditions.values.numpy(), preditions.indices.numpy()    

im = Image.open(args['image'])
test_image = np.asarray(im)
processed_test_image = process_image(test_image)

probs, classes = predict(processed_test_image, reloaded_keras_model, args['k_top'])
top_class_names = [class_names[str(i+1)] for i in classes[0]]

for i, j in zip(top_class_names, probs[0]):
    print('top class: {} with probabillity = {:.2f}%.'.format(i, j * 100))