from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import json
import base64
import cv2
import PIL
from PIL import Image

__model = None

def classify_image(image_base64, file_path=None):

    img = get_image(file_path,image_base64)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    classes = __model.predict(img_data)
    if classes[0, 0] == 0.0:
        return "Pneumonia"
    else:
        return "Normal"


def load_saved_artifacts():
    print("Loading Saved Artifacts....Start")
    global __model
    if __model is None:
        __model = load_model('./artifacts/model_vgg16.h5')
    print("loading saved artifacts...done")


def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_image(image_path, image_base64_data):
    if image_path:
        img = image.load_img(image_path,target_size=(224,224))
    else:
        img1 = get_cv2_image_from_base64_string(image_base64_data)
        img = cv2.resize(img1,(224,224))
    return img

def get_b64_test_image_for_normal():
    with open("b64.txt") as f:
        return f.read()

def get_b64_test_image_for_pneumonia():
    with open("b65.txt") as f:
        return f.read()


if __name__ == '__main__':
    load_saved_artifacts()

    print(classify_image(get_b64_test_image_for_normal(), None))
    print(classify_image(get_b64_test_image_for_pneumonia(), None))
    # print(classify_image(None, "./val/PNEUMONIA/person1947_bacteria_4876.jpeg"))
