"""
Memory
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import nltk
from nltk.tokenize import word_tokenize , sent_tokenize
import preprocess_kgptalkie as kgp
from DataStructures import FIFO

import keras
from keras.preprocessing.text import Tokenizer
from keras.saving.save import load_model
from keras import  callbacks
from keras.utils import pad_sequences
from keras.layers import concatenate
from keras.layers import Input , Embedding , CuDNNLSTM , Dense , GlobalMaxPooling1D , Flatten,\
    Bidirectional , SimpleRNN , LSTM , Concatenate
from keras.models import Model
from keras import losses , optimizers
from keras.utils import plot_model

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv('data/emotion_69k_preprocess2.csv')
"""
vocab_list = []
for sent in df['dialogue_in']:
    tokens = word_tokenize(sent)
    for token in tokens:
        if token not in vocab_list:
            vocab_list.append(token)

print(len(vocab_list))
# print(sorted(vocab_list))
"""
tokenizer = Tokenizer(num_words = 18000)
tokenizer.fit_on_texts(df['dialogue_in'])
train_in = tokenizer.texts_to_sequences(df['dialogue_in'])
train_out = tokenizer.texts_to_sequences(df['dialogue_out'])

#apply padding
X_train_in = pad_sequences(train_in)
X_train_out = pad_sequences(train_out , 220)

#encoding the outputs
le = LabelEncoder()
y_train = le.fit_transform(df['labels'])

#input length
input_shape_in = X_train_in.shape[1]
input_shape_out = X_train_out.shape[1]
#define vocabulary
vocabulary = len(tokenizer.word_index)
#output length
output_length = le.classes_.shape[0]

class CustomCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.98:
            print('Accuracy over 98%, Training ending')
            self.model.stop_training = True

def scheduler(epoch , lr):
    if epoch < 2:
        return lr
    else:
        return lr * 0.99

lr_scheduler = callbacks.LearningRateScheduler(scheduler , verbose = 1)


i = Input(shape = (input_shape_in,) , name = 'user_input')
x = Embedding(vocabulary + 1 , 10 , name = 'user_embedding')(i)
x = LSTM(10 , return_sequences = True , name = 'user_lstm')(x)


i2 = Input(shape = (input_shape_in,) , name = 'ai_input')
y = Embedding(vocabulary + 1 , 10 , name = 'ai_embedding')(i2)

concatted = Concatenate()([x , y])

x = Bidirectional(LSTM(10 , return_sequences = True),
                  backward_layer = LSTM(10 , return_sequences = True , go_backwards = True),
                  name = 'bidir_lstm')(concatted)
# x = SimpleRNN(16 , return_sequences = True , name = 'RNN')(x)
x = Flatten()(x)
x = Dense(output_length , activation = 'sigmoid' , name = 'output')(x)

model = Model(inputs = [i , i2] , outputs = x)

#compiling the model
model.compile(loss = losses.SparseCategoricalCrossentropy(),
              optimizer = optimizers.Adam(),
              metrics = ['accuracy'])

print(model.summary())
plot_model(model , 'model_logs/UbuntuModel_MkII.png',
           show_shapes = True,
           show_dtype = True,
           show_layer_names = True,
           show_layer_activations = True,)

#training the model
train = model.fit(x = [X_train_in , X_train_out] , y = y_train,
                  epochs = 20,
                  callbacks = [lr_scheduler])

np.save('model_logs/UbuntuModel_MkII_history' , train.history)
model.save('models/UbuntuModel_MkII')


# model = load_model('models/TFChatbot_wRNN_Mem')

text = 'hello'
text2 = ''
text = tokenizer.texts_to_sequences([text])
text2 = tokenizer.texts_to_sequences([text2])
text = np.array(text).reshape(-1)
text2 = np.array(text2).reshape(-1)
text = pad_sequences([text] , 220)
text2 = pad_sequences([text2] , 220)
text = model.predict([text , text2])
text = text.argmax()
print(f'BOT: {le.inverse_transform([text])}')

print(model.summary())

in_fifo = FIFO(3 , initial_state = ['' , '' , ''])
out_fifo = FIFO(3 , initial_state = ['' , '' , ''])

while text != 'done':
    text = input('USER: ')
    in_fifo.append(text)
    # Preprocessing
    text_in = tokenizer.texts_to_sequences([in_fifo.str_concat('. ')])
    text_out = tokenizer.texts_to_sequences([out_fifo.str_concat('. ')])

    text_in = np.array(text_in).reshape(-1)
    text_out = np.array(text_out).reshape(-1)

    text_in = pad_sequences([text_in] , 220)
    text_out = pad_sequences([text_out], 220)

    # Prediction
    pred = model.predict([text_in , text_out]) # test switching the inputs with the MkI model
    pred = pred.argmax()
    pred = le.inverse_transform([pred])[0]
    pred = pred.strip('"')

    print(f'BOT: {pred}')
    out_fifo.append(pred)
