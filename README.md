# ambhatik
Welcome to Ambhatik!
Batik pattern classifier using neural network ResNet50

Ambhatik is a website application that is fully built in using python.
We are aiming to train our machine to recognize and classify batik pattern in images. 

Data :We have selected four classes of batik pattern that we use in our model which are batik parang, batik kawung, batik megamendung, and batik sekar jagad in total of 199 images.
To build the model, we use selenium to crawl data from Google Images. Please note that we do not own any rights and do not claim any pictures we show and use here.

Algorithm: First we augment our data to get more variation and to prevent overfitting, then train the model, test the model, and review our model.
To get the best model we experiment with the learning rate. We save the weight of our best model by looking at the validation accuracy.

Evaluations
Avg. Validations Accuracy : 0.775
Avg. Precision : 0.76
Avg. Recall : 0.752
Avg. f-1 score : 0.74


Steps to Reproduce
1. download and extract files that will be included in the webapp on http://bit.ly/ambhatik
2. make sure you have all the packages used in the program (streamlit, keras, tensorflow, etc.)
3. on your python environment "streamlit run batik.py"
4. the webapp will run on your browser

We welcome anyone who wants to collaborate.
