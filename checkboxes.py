import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt 
import os
import random
import matplotlib.image as mpimg
import glob
from PIL import Image
from main_program import create_model, predict,\
     predict_upload, open_and_resize

#import SessionState

def checkbox_batik_types(batik_types):
    # checkbox 1 (Learn About batik types)
    cb_types = st.sidebar.checkbox("Tick to learn about batik types",
                               False, key='cb_types')
    if cb_types:
        st.markdown("---")
        batik_info = st.sidebar.selectbox("Batik Description", batik_types)

        if batik_info:
            img = glob.glob(".\\description\\" + str(batik_info) + "\\*.jpg")[0]  
            descr = glob.glob(".\\description\\" + str(batik_info) + "\\*.txt")[0]
            f = open(descr, 'r')
            st.image(open_and_resize(img),
                     caption=batik_info) 
            
            st.write(f.read())


def checkbox_machine_learning():
    st.sidebar.markdown("---")
    st.sidebar.markdown("Ambhatik Machine Learning")

    cb_ml = st.sidebar.checkbox("Tick to see machine predictions", False,
                                key='cb_ml')
    
    if cb_ml:
        ml_info = st.sidebar.selectbox("Machine Learning Details",
                                       ["Data","The Model","Model Evaluation",
                                        "Predict"])
        if ml_info == "Data":
            st.markdown("---")
            st.subheader("Training Data")
            st.write("""To build the model, we use selenium to crawl data from Google Images and sorted the image
                        for quality control, we then perform image augmentation before feeding the image into the network to prevent
                        overfitting. Below here you will find the number of data in each category and random sample images from our
                        training data. Please note that we do not own any rights and do not claim any pictures we show and use
                        here.""")
           
            data = pd.DataFrame({'Pattern':['kawung','megamendung','parang','sekar_jagad'],
                                 'Total':[47,44,50,58]})
            
            data.set_index('Pattern',inplace=True)
            st.bar_chart(data)

            # sample image
            n = st.slider("Choose how many sample images do you want to see",0,5)
            image_dir = ".\\Indonesia\\batik_categorized\\"
            batik_dirs = os.listdir(image_dir)
            
            for i in range(n):
                name = random.choice(batik_dirs)
                random_image = random.choice(os.listdir(os.path.join(image_dir, name)))
                random_image_file = os.path.join(image_dir,name,random_image)
                st.image(open_and_resize(random_image_file),
                         caption=name)

        elif ml_info == "The Model":
            st.markdown("---")
            st.subheader("Our Ambhatik Algorithm")

            conv_descr = open(".\\algorithm\\convnet_description.txt", 'r')
            res_descr = open(".\\algorithm\\resnet_description.txt", 'r')
            st.write("Here we will give you a brief introduction of the algorithm that we use in this project.")

            st.write(conv_descr.read())
            st.image(open_and_resize(".\\algorithm\\conv_img.jpeg", height=300))
            
            st.write(res_descr.read())
            st.image(open_and_resize(".\\algorithm\\res_img.png", height=200))
            
        elif ml_info == "Model Evaluation":
            st.markdown("---")
            st.subheader("Model Evaluation")
            st.write("Here we will show the result of our training, confusion matrix, the result of validation test")
            st.image(open_and_resize(".\\algorithm\confusion_matrix.png", height=500))

            # create dataframe
            df = pd.DataFrame({'precision' : [0.64, 1.00, 0.60, 0.8],
                               'recall' : [0.88, 0.86, 0.38, 0.89],
                               'f1-score' : [0.74, 0.92, 0.46, 0.84],
                               'support' : [8, 7, 8, 9]},
                              index=['Batik Kawung', 'Batik Megamendung',
                                     'Batik Parang', 'Batik Sekarjagad'])

            st.dataframe(df)
            
        elif ml_info == "Predict":
            st.markdown("---")
            st.subheader("Try your own!")
            st.write("Upload batik image and the machine will try to guess which family the batik belongs to")

            model = create_model()
            # predict uploaded batik image
            predict_upload(model)


def checkbox_games():
    st.sidebar.markdown("---")
    st.sidebar.markdown("Let's Play Game!")

    cb_game = st.sidebar.checkbox("Tick to compete with the machine",
                                  False, key='cb_game')
    
    if cb_game:
        ml_game = st.sidebar.selectbox("Ambhatik Games",["Start the Game"])
##        if ml_game == "Rules":
##            st.markdown("---")
##            st.subheader("Are you smarter than the machine?")
##            st.write("Instructions:")
##            st.write("1. Broaden your knowledge by reading 'Learn about individual batik' section")
##            st.write("2. You and the machine will have 5 times to guess the batik pattern shown in the image")
##            st.write("3. After 5 rounds please head to 'Who Wins?'section")
##            st.write("4. Have fun and enjoy the game!")
##            st.markdown("GOOD LUCK!")

        if ml_game == "Start the Game":
            st.markdown("---")
            st.subheader("Guess the Pattern!")
            st.write("Hint: It is either Kawung, Parang, Megamendung, or Sekarjagad")

            #test_image_file = ".\\images_data\\batik_test\\keris\\MG_5087.jpg"

            image_file = glob.glob(".\\images_data\\batik_test\\keris\\*.jpg")
            test_image_file = image_file[0]
            
            st.image(open_and_resize(test_image_file))

            model = create_model()
            prob, label = predict(test_image_file, model)

            # text input placeholder
            ti_placeholder = st.empty()

            val = " "
            title = ti_placeholder.text_input('Guess the batik pattern!',
                                              value=val)

            title
            if title != val:
                st.write("Machine guessed", label)
                st.write('You guessed', title)
                

##        elif ml_game == "Who Wins?":
##            st.markdown("---")
##            st.write("The winner is....... (Kendang Roll)")
##            st.write("MACHINE!")
##            st.write("YOU!")            
