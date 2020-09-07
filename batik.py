import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt 
import os
import random
import matplotlib.image as mpimg
import glob
from PIL import Image
from main_program import create_model
from checkboxes import checkbox_batik_types, checkbox_machine_learning,\
     checkbox_games

#from IPython.display import display, Image

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def header():
    st.header("Welcome to Ambhatik!")
    st.write("A place to learn about batik, the world heritage from Indonesia, and predict batik pattern using machine learning.")

def sideBar():
    st.sidebar.title("Ambhatik Contents")
    st.sidebar.markdown("Learn about individual batik")
    dirs = glob.glob(".\\description\\*")
    batik_types = [os.path.split(dirs)[1] for dirs in dirs]

    # checkbox 1 (Learn About batik types)
    checkbox_batik_types(batik_types)
    
    # checkbox 2
    checkbox_machine_learning()

    # checkbox 3
    checkbox_games()       
			
def main():
    header()
    sideBar()

if __name__ == "__main__":   
    main()
