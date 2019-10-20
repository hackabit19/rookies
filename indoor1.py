from __future__ import absolute_import, division, print_function, unicode_literals


import requests
import json
import nltk

import cv2

import tensorflow as tf


import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.applications import InceptionV3
import tensorflow as tf 

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle

from tensorflow.python.framework import ops
ops.reset_default_graph()


##nltk.download('stopwords')
##stopwords = nltk.corpus.stopwords.words('english')
import pyttsx3 as pyttsx

annotation_file = 'annotations/captions_train2014.json'

PATH = 'train2014/'

imgpath = 'test.jpg'

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path



        
##    print((i['confidence']),60)

"""model"""

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, 64, hidden_size)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    # attention_weights shape == (batch_size, 64, 1)
    # you get 1 at the last axis because you are applying score to self.V
    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

engine = pyttsx.init()
import requests

def c(img):
    cv2.imwrite("rtest.jpg",img)
    r = requests.post(
        "https://api.deepai.org/api/densecap",
        files={
            'image': open("rtest.jpg", 'rb'),
        },
        headers={'api-key': 'f693c409-092c-4948-b0b3-516f9c85f833'}
    )

    x=r.json()
    ##x=json.loads(x)
    s=''
    d={}
    for i in x['output']['captions']:
        if float(i['confidence'])*100>60:
            return i['caption']
            

class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


def indoor():
    vid = cv2.VideoCapture(0)
    count =0
    engine.setProperty('rate', 250)
    while True :
        count+=1
        ret,image = vid.read()
##        print(ret)
        cv2.imshow("df",image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            from main1 import main
            main()
            break
        
        if count%15==0:
            captions =c(image)
            print(captions)
            engine.say(captions)
            engine.runAndWait()

        
        
indoor()


