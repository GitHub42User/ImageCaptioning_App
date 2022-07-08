import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import pickle
import requests, zipfile, io
import os, shutil
from random import *

def Load_Images():
  r = requests.get("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip", stream=True)
  print(r.ok)
  z = zipfile.ZipFile(io.BytesIO(r.content))
  z.extractall("./Flickr")
  
  
features = pickle.load(open("images1.pkl", "rb"))
model = load_model('model_9.h5')
words_to_index = pickle.load(open("words.pkl", "rb"))
index_to_words = pickle.load(open("words1.pkl", "rb"))
  
  
def Make_Folder():
  newpath = 'Images/' 
  if not os.path.exists(newpath):
    os.makedirs(newpath)

  source_dir = 'Flickr/Flicker8k_Dataset'
  target_dir = 'Images/'
    
  file_names = os.listdir(source_dir)
    
  for file_name in file_names:
     shutil.move(os.path.join(source_dir, file_name), target_dir)

  folder = 'Flickr'
  for filename in os.listdir(folder):
     file_path = os.path.join(folder, filename)
     try:
          if os.path.isfile(file_path) or os.path.islink(file_path):
             os.unlink(file_path)
          elif os.path.isdir(file_path):
              shutil.rmtree(file_path)
     except Exception as e:
          print('Failed to delete %s. Reason: %s' % (file_path, e))
        
if not os.path.exists("Images"):
    Load_Images()
    Make_Folder()
    
images = "Images/"
max_length = 33


def Image_Caption(picture):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [words_to_index[w] for w in in_text.split() if w in words_to_index]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([picture,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_words[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

st.title('Image Captioning Group 13')

with st.expander("Introduction"):
  st.header("Introduction")
  st.write("Hello, we are Leon Lang, Jean Louis Fichtner and Loredana Bratu and we created this app as a part of our business informatics course. On this page you will have the opportunity to see how we developed step by step this project, what was our motivation and is our future vision.")
  st.write("But before starting, we have a question:  What do you see in the picture bellow?")
  st.image("https://futalis.de/wp-content/uploads/2020/07/contentbild-hund-fital-1.jpg")
  
with st.expander("Solution"):
  st.header("Solution")
  st.write("Most probably you would say “A dog running on a field”, some may say “a dog with white and black spots” and others might say “a dog on a grass and some yellow flowers”.")
  st.write("Definitely all of these captions are correct for this image and there may be some more as well. But the point we want to make is that it is so easy for us, as human beings, to just have a look at a picture and describe it in an appropriate language.")
  st.write("But, can you write a AI based computer program that takes an image as input and produces a relevant caption as output?")
  st.image("https://img.freepik.com/premium-vector/cute-funny-think-emoji-smile-face-with-question-mark-vector-flat-line-doodle-cartoon-kawaii-character-illustration-icon-isolated-white-background-yellow-emoji-circle-think-character-concept_92289-3170.jpg")
  
  
with st.expander("Business Understanding"):
  st.header("Business Understanding")
  st.write("The upper question is what also determined us to analyse this problem and to develop a corresponding solution to it.")
  st.image("https://iq.opengenus.org/content/images/2020/06/Machine-Caption.png")
  st.subheader("Motivation")
  st.write("First, we tried to understand how important this problem is to real world scenarios. Let’s see few applications where a solution to this problem can be very useful:")
  st.write("Self-driving cars -Automatic driving is one of the biggest challenges and if we can properly caption the scene around the car, it can give a boost to the self-driving system.")
  st.image("https://www.autonomousvehicleinternational.com/wp-content/uploads/2019/05/1.1-For-CapGemini-story.gif")
  st.write("Support the blinds - We can create a product for the blind which will guide them travelling on the roads without the support of anyone else. We can do this by first converting the scene into text and then later the text to voice.")
  st.image("https://media.istockphoto.com/photos/blind-man-crossing-a-street-picture-id1292075242?k=20&m=1292075242&s=612x612&w=0&h=CeztC1Y2erLKVZrnI1JoUf7DfskR9D1bb8933ul023w=")
  st.write("Monitoring cameras-we can also generate relevant captions, then we can raise alarms as soon as there is some malicious activity going on somewhere. This could probably help reduce some crime and/or accidents.")
  st.image("https://i.tribune.com.pk/media/images/552016-cameracctvsecurity-1369077101/552016-cameracctvsecurity-1369077101.jpg")
  st.write("•	Automatic Captioning Apps - can help one to explore unknown things around you using only your phone’s camera which scans a real object converting that instant photo into a caption and telling you what this might be (example Google Lens).")
  st.image("https://storage.googleapis.com/gweb-uniblog-publish-prod/images/lens_dog.max-1000x1000.png")
  st.subheader("Goal")
  st.write("Our main goal is our app to automatically generate captions, also known as textual descriptions, for random images. The dataset will be in the form [image → captions]. It will consist of input images and their corresponding output captions which have to be as precise as possible, but also short and concise. The caption generator will involve the dual techniques from computer vision - to first understand the content of the image, and a language model from the field of natural language processing to turn the understanding of the image into words in the right order and correct structure.")
  st.subheader("Project Plan")
  st.write("For a better overview and structure to our project, we followed the next steps:")
  st.image("https://files.gitbook.com/v0/b/gitbook-legacy-files/o/assets%2F-MUgl8cG7yNhLCPnDv5x%2F-MY4KgeUGKq9SC1IR8Kr%2F-MY4PAID8A_U-333h3JC%2Fimage.png?alt=media&token=1082c6fe-0617-41df-b8f5-d71552200417")
  

with st.expander("Data Understanding"):
  st.header("Data Understanding")
  st.subheader("About or data...")
  st.write("Our dataset consists of over 80000 images with at least 5 captions each, which are from the open-source dataset MS COCO and are randomly sorted. For the training we will not use the whole data set, because it has an enormous storage capacity.")
   st.write("Our dataset consists of over 80000 images with at least 5 captions each, which are from the open-source dataset MS COCO and are randomly sorted. For the training we will not use the whole data set, because it has an enormous storage capacity.")
  
  st.code('''
  #input
 data['1000268201_693b08cb0e']
  
 #output
 ['A child in a pink dress is climbing up a set of stairs in an entry way .',
 'A girl going into a wooden building .',
 'A little girl climbing into a wooden playhouse .',
 'A little girl climbing the stairs to her playhouse .',
 'A little girl in a pink dress going into a wooden cabin .']
 
  ''')
  st.write("Therefore, in advantage of time and costs we only used ca. 8.000 pictures to train our data.")
  st.text("")
  st.write("We will not be delimiting our dataset in specific domains, as our purpose is not image classification, but image captioning, so it’s in our best interests to vary the image topics, so that we call achieve a high accuracy.")
  
with st.expander("Data Preperation"):
  st.header("Data Preperation")
  
  st.subheader("Data Cleaning")
  st.write("Our first step is to preprocess the dataset and prepare the captions by making some necessary changes. For example, we will make sure that all words are converted to lowercase, as we do not want the same word to be stored as two different vectors when computing the problem. We will also remove words less than two in length to ensure that we remove irrelevant characters such as single letters and punctuation marks. The function and code to solve this task are as follows:") 
  st.code('''
  
# Cleanse and pre-process the data

def cleanse_data(data):
    dict_2 = dict()
    for key, value in data.items():
        for i in range(len(value)):
            lines = ""
            line1 = value[i]
            for j in line1.split():
                if len(j) < 2:
                    continue
                j = j.lower()
                lines += j + " "
            if key not in dict_2:
                dict_2[key] = list()
            
            dict_2[key].append(lines)
            
    return dict_2

data2 = cleanse_data(data)
print(len(data2))    ''')
  
  
  st.write("In our next step, we will create a dictionary to store the image files and their respective captions accordingly. We know that each image has an option of five different captions to choose from. We will define the .jpg image as the key with their five respective captions representing the values. We will split the values appropriately and store them in a dictionary. The following function can be written as follows:")
  st.code(''' 

def vocabulary(data2):
    all_desc = set()
    for key in data2.keys():
        [all_desc.update(d.split()) for d in data2[key]]
    return all_desc

# summarize vocabulary
vocabulary_data = vocabulary(data2)
print(len(vocabulary_data)) ''')
  st.subheader("Pre-Processing the Images")
  st.write("Further we have used the Inception V3 transfer learning model to convert each of the respective images into their fixed vector size. The model makes use of the pre-trained weights on the image net to achieve the computation of the following task with relative ease. Once we finished the computation of pre-processing of the images, we have saved all these values in a pickle file that helped us to utilize these models separately during the prediction process. This process can be completed with the following code block:")
  st.code('''
images = 'Images/'
img = glob(images + '*.jpg')
print(len(img))

def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = keras.preprocessing.image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # pre-process the images using preprocess_input() from inception module
    x = keras.applications.inception_v3.preprocess_input(x)
    return x
  ''')
  
  
with st.expander("Data Modeling"):
  st.header("Data Modeling")
  st.write("The modeling technique we used is the CNN - Convolutional Neural Network within Deep Learning which is a type of artificial neural network that is widely used for image/object recognition and classification. The encoder-decoder architecture - where an input image is encoded into an intermediate representation of the information contained within the image and subsequently decoded into a descriptive text sequence - has also contributed to caption’s generation.")
  
  st.subheader("Model architecture")
  st.write("Finally, let us define the LSTM and embedding layers to create our model architecture. For this model, we will make use of two inputs, namely the image vectors and the word embeddings of the captions, for making the predictions. The embedding vector is passed through an LSTM architecture which will learn how to make the appropriate word predictions. The image vector and the LSTM predictions are combined together as one unit and passed through some dense layers and a SoftMax activation function, which is equivalent to the size of the pre-defined vocabulary. ")
  st.code(''' 
  embeddings_index = {} # empty dictionary
f = open('glove.6B.200d.txt', encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 200

# Get 200-dim dense vector for each of the 10000 words in out vocabulary
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in wordtoix.items():
    #if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros,
        embedding_matrix[i] = embedding_vector
        
embedding_matrix.shape

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.summary()

  ''')
  st.subheader("Model training")
  st.write("We trained our Model over 35 Epochs and with 3 pics per bath. We used a categorical cross-entropy loss function and the Adam optimizer.")
  st.code('''
  model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam')

epochs = 35
number_pics_per_bath = 3
steps = len(data2)//number_pics_per_bath

features = pickle.load(open("images1.pkl", "rb"))

# https://stackoverflow.com/questions/58352326/running-the-tensorflow-2-0-code-gives-valueerror-tf-function-decorated-functio

tf.config.run_functions_eagerly(True)

for i in range(epochs):
    generator = data_generator(data2, features, wordtoix, max_length, number_pics_per_bath)
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save('model_' + str(i) + '.h5')
  ''')
  
  st.subheader("Model testing")
  st.write("The respective function that will take in the images, load their vectors, create word embedding, and utilize the saved model for making the appropriate predictions.")
  st.code('''
  features = pickle.load(open("images1.pkl", "rb"))
model = load_model('model_9.h5')
images = "Images/"
max_length = 33
words_to_index = pickle.load(open("words.pkl", "rb"))
index_to_words = pickle.load(open("words1.pkl", "rb"))

def Image_Caption(picture):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [words_to_index[w] for w in in_text.split() if w in words_to_index]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([picture,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_words[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final
  ''')  
  

with st.expander("Evaluation"):    
   st.subheader("Evaluation")
   st.write("Before proceeding to final deployment of the model, it is important to evaluate the model, and review the steps executed to construct the model, to be certain it properly achieves the business objectives. A key objective is to determine if there is some important business issue that has not been sufficiently considered.")
   st.text("")
   st.write("")
   st.image("https://blog.forecast.it/hs-fs/hubfs/accuracy-precision.jpg?width=454&name=accuracy-precision.jpg&w=144")
   st.write("Accuracy: The ability of a measurement device to match the actual value of the quantity being measured.")
   st.write("Precision: The reproducibility of repeated measurements and a measure of their likely spread.")
   st.subheader("Our Difficulties")
   st.write("We evaluated our program based on the Accuracy/Precision model and we came to the conclusion that our model achieved a high Precision, but a moderate Accuracy. In the majority of cases the app can easily recognize what is in the picture, but unfortunately it does not describe the picture sufficiently enough. We also noticed a few more problems: Sometimes the sentence is not ended or the last word is left out. In addition, it often has problems to tell colors or genders apart.")
    
    
    
with st.expander("Our Image Captoning Program"): 
  st.write("You can try our Program with a random image from our preparated dataset by pressing the following Button.")
  if st.button('random picture'):
       Variable = [500, 910, 3273, 1863, 1566, 1386, 1355, 1824, 728, 3856, 1453, 763, 1876, 765, 2766, 3041, 3041, 2440, 2286, 4211, 4200, 4502, 4526, 1326, 4122, 1927, 2137, 473, 4489, 3947]
       y = randint(0, 29)
       z = Variable[y]
       pic = list(features.keys())[z]
       image = features[pic].reshape((1,2048))
       x = plt.imread(images+pic)
       st.image(x)
       st.write("Caption:", Image_Caption(image))
