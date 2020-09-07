import numpy as np
import streamlit as st
import pickle
##from datagen import create_datagen, report
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from architecture import ResNet50
from PIL import Image

def create_model():
    """
    define our resnet 50 model and load pickled weight
    """
    # define model arcitecture
    model = ResNet50(input_shape=(224, 224, 3),
                     classes=4)
    # load weights
    with open('..\\batik_classifier\\weights.pkl', 'rb') as pkl:
        w = pickle.load(pkl)
        
    # set weights
    model.set_weights(w)
    
    return model

def open_and_resize(path, height=400):
    """
    resize image to widthx400 by keeping its
    aspect ratio constant

    :params path:
        input image directory

    returns pillow image object
    """
    img = Image.open(path)
    wpercent = (height / img.size[1])
    vsize = int(img.size[0] * wpercent)
    img = img.resize((vsize, height), Image.ANTIALIAS)
    return img


def pil2array(img):
    """
    convert pillow image object to a numpy array
    to be fed to the neural network for prediction, training,
    or validation

    :params img:
        image path / directory
    """
    img = Image.open(img)
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = np.array(img)
    return img/255.

    
def predict(image_path, model):
    """
    predict batik types given the model and image,
    returns predicted probability and batik types

    :params image_path:
        image directory you wish to predict
    :params model:
        our resnet50 model

    """
    class_name = ['Batik Kawung', 'Batik Mega Mendung',
                  'Batik Parang', 'Batik Sekar Jagad']
    
    # convert image to np array
    image_pred = pil2array(image_path)
    image_pred = np.expand_dims(image_pred, axis=0)

    prediction = model.predict(image_pred)
    probability = prediction[0][np.argmax(prediction)] * 100
    pred_class = class_name[np.argmax(prediction)]
    return probability, pred_class


def predict_upload(model):
    """
    create streamlit file uploader widgets, and returns
    predicted batik types using specified model

    :params model: our custom ResNet50 model
    """
    
    st.title("Batik classification")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    uploaded_file = st.file_uploader("Choose an image..", type=['jpg', 'png'])

    if uploaded_file is not None:
        #image = Image.open(uploaded_file)
        st.image(open_and_resize(uploaded_file), caption='Uploaded Image.')
        st.write("")
        st.write("Classifying..")
        prob, label = predict(uploaded_file, model)
        st.write(f"The machine is {prob:.2f} % sure that this is a family of {label}")



#### Test validation
##
##    _, valid_images, _, step_valid = create_datagen(input_shape=(224, 224, 3),
##                                                    split=0.3,
##                                                    path=".\\images_data")
##    report(model, valid_images, step_valid)





    
    

    
