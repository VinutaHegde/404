# -*- coding: utf-8 -*-
"""Image Captioning

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1V0WvawN5uJykeFCs6PN77P0wpugEFker
"""

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth, files
from oauth2client.client import GoogleCredentials

from keras.layers import Embedding, Input, GRU, Dense, Dropout
from keras.utils import plot_model, Sequence
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd
import json
import random
import csv
import os
import zipfile

SEED = 10
IMAGE_EMBEDDING_DIM = 2048
COMPRESSED_HIDDEN_DIM = 1280
EMBEDDING_DIM = 300

BATCH_SIZE = 64
EPOCHS = 10

"""#Load data"""

def load_data(file_id, file_name):
  """
  Generic function to import file with 'file_id' with 'file_name'.
  """  
  auth.authenticate_user()
  gauth = GoogleAuth()
  gauth.credentials = GoogleCredentials.get_application_default()
  drive = GoogleDrive(gauth)

  if not os.path.exists("./data"):
    os.mkdir("data")

  handle = drive.CreateFile({'id': file_id})
  handle.GetContentFile('data/' + file_name)

  return

load_data('1NpLC7ApLS32X0ZHxmbhRmBNKZhNtO8aV', 'labels.json')
load_data('1PqAbS3dSf7qdwYHD9fuH8aAt4eONpnlX', 'glove.6B.300d.txt')
load_data('1kAvj8Yfu65Ut-I5P9DXx4CRjnsAXfwm_', 'image_embeddings.json')
load_data('1ZGnhD3uH5mJqAC6aXdfzOPBKWI4_DdYg', 'image_to_captions.csv')
load_data('1uAzvohxDxBuIQ7q32jVp6LTckfPwQrtT', 'all_captions.json')

"""# Data Preparation"""

with open("data/all_captions.json", "r") as handle:
  all_captions = json.load(handle)

dataset = []

for album_id, album_values in all_captions.items():
  for story_id, story_values in album_values.items():

    images_in_story = []
    caption_of_images = []

    for image_id, caption in story_values:

      images_in_story.append(image_id)
      caption_of_images.append(caption)

    row = {"image_ids" : images_in_story, 
           "captions" : " ".join(caption_of_images)}

    dataset.append(row)

dataset = pd.DataFrame(dataset)

del all_captions

"""# Preprocess Images"""

load_data('1GpcwnFqNqZWUgfQ-LUrLvnCgg-fPrpyt', 'embedding_chunks.zip')

archive = zipfile.ZipFile("data/embedding_chunks.zip")
archive.extractall("data/embedding_chunks")

def get_image_embedding(no_chunks, directory):

  image_embedding = {}

  for i in range(no_chunks):
    file_name = directory + '/cnn_group'+ str(i+1) + '.json'
    with open(file_name) as json_file:
      json_data = json.loads(json.load(json_file))
      image_embedding = {**image_embedding, **json_data}
        
  return image_embedding

image_embeddings = get_image_embedding(10, 'data/embedding_chunks')
image_ids = image_embeddings.keys()

stories = [[image if image in image_ids else '' for image in story] for story in dataset["image_ids"]]
#indices = [images != ['']*5 for images in available_images]

UNK_IMG = np.zeros(IMAGE_EMBEDDING_DIM)
story_input = np.zeros((len(stories), len(stories[0]) * IMAGE_EMBEDDING_DIM))

for index, story in enumerate(stories):
  story_input[index] = np.concatenate([UNK_IMG if image == '' else image_embeddings[image] for image in story])

HIDDEN_DIM = story_input.shape[1]

del image_embeddings, image_ids

"""# Preprocessing Captions"""

import matplotlib.pyplot as plt

tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset["captions"])
sequences = tokenizer.texts_to_sequences(dataset["captions"])
length = pd.Series([len(s) for s in sequences])

print(length.describe())

length.plot.box(vert=False)
plt.subplots_adjust(left=0.25)
plt.show()

def tokenize_captions(captions, max_length=75):
  
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(captions)
  sequences = tokenizer.texts_to_sequences(captions)
  word_index = tokenizer.word_index
  #sequences = [s for s in sequences if len(s) < max_length]
  #data = pad_sequences(sequences, maxlen=max([len(s) for s in sequences]))
  data = pad_sequences(sequences, maxlen=max_length)

  return data, word_index

def encode_captions(captions):
  
  captions_input = ["<sos> " + x for x in captions]
  captions_output = [x + " <eos>" for x in captions]

  captions_input, input_wtoidx = tokenize_captions(captions_input)
  captions_output, output_wtoidx = tokenize_captions(captions_output)

  return np.array(captions_input), np.array(captions_output), input_wtoidx, output_wtoidx

(captions_input, 
 captions_output, 
 input_wtoidx, 
 output_wtoidx) = encode_captions(dataset["captions"])

def get_embedding_matrix(filename, word_to_index, embedding_dim):
  embeddings_index = {}

  with open(filename) as f:

    for index, line in enumerate(f):  
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs

  embedding_matrix = np.zeros((len(word_to_index)+1, embedding_dim))

  for word, index in word_to_index.items():

    if word in embeddings_index:
      embedding_vector = embeddings_index[word]
      embedding_matrix[index] = embedding_vector

  return embedding_matrix

embedding_matrix = get_embedding_matrix('data/glove.6B.300d.txt', input_wtoidx, 300)

del dataset

"""# Data Generator"""

class DataGenerator(Sequence):
        
    def __init__(self, X, Y_in, Y_out, batch_size=64, shuffle=True):

        self.X = X
        self.Y_in = Y_in
        self.Y_out = Y_out

        #Hyperparameters      
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.X.shape[0] / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        X, Y_in, Y_out = self.__data_generation(indices)

        #expand_dims() to handle https://github.com/tensorflow/tensorflow/issues/17150
        return [[Y_in, X], np.expand_dims(Y_out, 2)]

    def on_epoch_end(self):
        self.indices = np.arange(self.X.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, indices):

        X = self.X[indices]
        Y_in = self.Y_in[indices]
        Y_out = self.Y_out[indices]

        return X, Y_in, Y_out

def split_dataset(X, Y_in, Y_out, split_ratio):

  mask = np.random.randn(X.shape[0]) < split_ratio
  X_train, Y_in_train, Y_out_train = X[mask], Y_in[mask], Y_out[mask]
  X_test, Y_in_test, Y_out_test = X[~mask], Y_in[~mask], Y_out[~mask]
  
  return X_train, Y_in_train, Y_out_train, X_test, Y_in_test, Y_out_test

(X_train, 
 Y_in_train, 
 Y_out_train, 
 X_test, 
 Y_in_test, 
 Y_out_test) = split_dataset(story_input, captions_input, captions_output, 0.8)


(X_train, 
 Y_in_train, 
 Y_out_train, 
 X_val, 
 Y_in_val, 
 Y_out_val) = split_dataset(X_train, Y_in_train, Y_out_train, 0.8)

del story_input, captions_input, captions_output

"""#Model"""

def image_caption_model(embedding_matrix,
                        word_to_index, 
                        max_length, 
                        embedding_dim=300, 
                        hidden_dim=10240,
                        compressed_hidden_dim = 1280,
                        output_dim=28554):
    
    hidden_state = Input(shape=(hidden_dim, ), dtype='float32', name="Hidden")
    dropped_hidden_state = Dropout(0.5) (hidden_state)
    compressed_hidden_state = Dense(compressed_hidden_dim, activation='relu', name='Compress_hidden')(dropped_hidden_state)

    inputs = Input(shape=(max_length, ), dtype='int32', name="Inputs")

    embedding_layer = Embedding(len(word_to_index)+1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_length,
                                trainable=False,
                                name="Embedding")
    embeddings = embedding_layer(inputs)

    decoder = GRU(compressed_hidden_dim, 
                  return_state=True, 
                  return_sequences=True, 
                  recurrent_dropout=0.5,
                  name="Decoder")
    
    outputs, _ = decoder(embeddings, initial_state=compressed_hidden_state)
    
    dropped_outputs = Dropout(0.5) (outputs)
    outputs = Dense(output_dim, activation='softmax', name='Dense')(dropped_outputs)

    model = Model(inputs=[inputs, hidden_state], outputs=outputs)

    return model

model = image_caption_model(embedding_matrix, 
                            input_wtoidx, 
                            max_length=75, 
                            embedding_dim=EMBEDDING_DIM, 
                            hidden_dim=HIDDEN_DIM,
                            output_dim = len(output_wtoidx)+1)

model.summary()

model.compile(optimizer='Adam',
              metrics=['accuracy'],
              loss='sparse_categorical_crossentropy')

from keras.utils import plot_model
plot_model(model)

indices = np.random.choice(np.arange(X_train.shape[0]), size=2000, replace=False)
X_train, Y_in_train, Y_out_train = X_train[indices], Y_in_train[indices], Y_out_train[indices]

#Reduce the data size for now
!mkdir output

train_generator = DataGenerator(X_train, Y_in_train, Y_out_train)
val_generator = DataGenerator(X_val, Y_in_val, Y_out_val)

checkpoint = ModelCheckpoint('output/baseline_model', monitor='val_loss', verbose=1, save_best_only=True)
history = model.fit_generator(train_generator, 
                    epochs=EPOCHS, 
                    verbose=1,
                    validation_data=val_generator,
                    callbacks=[checkpoint],
                    shuffle=True)

files.download('output/baseline_model')

import matplotlib.pyplot as plt 

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()
plt.savefig('Accuracy.png')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()
plt.savefig('Loss.png')

files.download('Accuracy.png')
files.download('Loss.png')

"""# Inference"""

load_data('1sHjhmtVd7AWb8m_Zb4M88pnsXHnAKXcL', 'baseline_model')

inference = image_caption_model(embedding_matrix, 
                            input_wtoidx, 
                            max_length=75, 
                            embedding_dim=EMBEDDING_DIM, 
                            hidden_dim=HIDDEN_DIM,
                            output_dim = len(output_wtoidx))

inference.compile(optimizer='Adam',
              metrics=['accuracy'],
              loss='sparse_categorical_crossentropy')

inference.load_weights('data/baseline_model', by_name=True)

np.argmax([y.max() for y in Y_in_test])

np.delete(Y_in_test, 10631, 0)
np.delete(Y_out_test, 10631, 0)

test_generator = DataGenerator(X_test, Y_in_test, Y_out_test)
test_loss = inference.evaluate_generator(test_generator, verbose=1)

test_loss

indices = np.random.choice(np.arange(X_test.shape[0]), size=20, replace=False)
X_test, Y_in_test, Y_out_test = X_test[indices], Y_in_test[indices], Y_out_test[indices]

#https://github.com/keras-team/keras/issues/12586
predictions = inference.predict(x=[Y_in_test, X_test],
                                batch_size=BATCH_SIZE,
                                verbose=1)

# Beam search decoder
from math import log
 
def beam_search_decoder(data, k):

  sequences = [[[], 1.0]]

  for row in data:
    all_candidates = []

    for i in range(len(sequences)):
      seq, score = sequences[i]

      for j in range(len(row)):
        candidate = [seq + [j], score * -log(row[j])]
        all_candidates.append(candidate)

    ordered = sorted(all_candidates, key=lambda x: x[1])
    sequences = ordered[:k]

  return sequences

beam_size = 5
caption_indices = []

for row in predictions:
  result = beam_search_decoder(row, 5)
  caption_indices.append(result[0])

