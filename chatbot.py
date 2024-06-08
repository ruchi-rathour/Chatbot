#!/usr/bin/env python

import pandas as pd
import numpy as np
import joblib
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from pathlib import Path
import string
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder



from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



df = pd.DataFrame(columns=['question','labels'])
f = open('Downloads/intents.json')
data = json.load(f)
data = data['intents']
for dt in data:
    for quest in dt['patterns']:
        ap = pd.Series([quest,dt['tag']], index = df.columns)
#         print(ap)
        df = df.append(ap,ignore_index=True)



dfs = pd.DataFrame(columns=['response','labels'])
f = open('Downloads/intents.json')
data = json.load(f)
data = data['intents']
for dt in data:
    for quest in dt['responses']:
        ap = pd.Series([quest,dt['tag']], index = dfs.columns)
#         print(ap)
        dfs = dfs.append(ap,ignore_index=True)



lemmatizer = WordNetLemmatizer()
vocab = Counter()
labels = []

for entry in df['question']:
#     print(entry)
    tokens = entry.split()
#     print(tokens)
    punc_remove = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [punc_remove.sub('',w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    tokens = [word.lower() for word in tokens if len(word) > 1]
    vocab.update(tokens)
#     print(tokens)
joblib.dump(vocab,'vocab.pkl')



no_stop = []
for entry in df['question']:
    tokens = entry.split()
#     print(tokens)
    punc_remove = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [punc_remove.sub('',w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    tokens = [word.lower() for word in tokens if len(word) > 1]
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if word not in stop_words]
    joblib.dump(tokens,'tokens.pkl')
    no_stop.append(' '.join(tokens))
df['question'] = no_stop


test_list = list(df.groupby(by='labels',as_index=False).first()['question'])


test_index = []
for i,_ in enumerate(test_list):
    idx = df[df.question == test_list[i]].index[0]
    test_index.append(idx)


train_index = [i for i in df.index if i not in test_index]


' '.join(list(vocab.keys()))

def encoder(df,feature):
    t = Tokenizer()
    entries = [entry for entry in df[feature]]
    t.fit_on_texts(entries)
    joblib.dump(t,'tokenizer_t.pkl')
    vocab_size = len(t.word_index) + 1
    entries = [entry for entry in df[feature]]
    max_length = max([len(s.split()) for s in entries])
    encoded = t.texts_to_sequences(entries)
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded, vocab_size


X,vocab_size = encoder(df,'question')
df_encoded = pd.DataFrame(X)
df_encoded['labels'] = df.labels
df_encoded.head()


for i in range(0,2):
    dt = [0]*16
    dt.append('confused')
    dt = [dt]
    df_encoded = df_encoded.append(pd.DataFrame(dt).rename(columns = {16:'labels'}),ignore_index=True)


train_index.append(87)
test_index.append(88)



labl = LabelEncoder().fit_transform(df_encoded.labels)


mapper = {}
for index,key in enumerate(df_encoded.labels):
    if key not in mapper.keys():
        mapper[key] = labl[index]

dfs.labels = dfs.labels.map(mapper).astype({'labels': 'int32'})
dfs


dfs.to_csv('response.csv',index=False)


train = df_encoded.loc[train_index]
test = df_encoded.loc[test_index]



X_train = train.drop(columns=['labels'],axis=1)
y_train = train.labels
X_test = test.drop(columns=['labels'],axis=1)
y_test = test.labels



y_train =pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values
max_length = X_train.shape[1]
output = 17



early_stopping = EarlyStopping(monitor='val_loss',patience=10)
checkpoint = ModelCheckpoint("model-v1.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, verbose = 1, min_delta = 0.0001)
callbacks = [early_stopping,checkpoint,reduce_lr]


def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size,300, input_length=max_length))
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
    model.add(MaxPooling1D(pool_size=8))
    model.add(Flatten())
    model.add(Dense(17, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])    
    # summarize defined model
    model.summary()
    return model

model = define_model(vocab_size, max_length)



history = model.fit(X_train, y_train, epochs=500, verbose=1,validation_data=(X_test,y_test),callbacks=callbacks)


[np.argmax(i) for i in model.predict(X_test)][:10]
[np.argmax(i) for i in y_test][:10]


from tensorflow.keras.models import load_model
model = load_model('model-v1.h5')
tokenizer_t = joblib.load('tokenizer_t.pkl')
vocab = joblib.load('vocab.pkl')


def get_pred(model,encoded_input):
    pred = np.argmax(model.predict(encoded_input))
    return pred


def bot_precausion(df_input,pred):
    words = df_input.question[0].split()
    if len([w for w in words if w in vocab])==0 :
        pred = 1
    return pred

def get_response(df2,pred):
    upper_bound = df2.groupby('labels').get_group(pred).shape[0]
    r = np.random.randint(0,upper_bound)
    responses = list(df2.groupby('labels').get_group(pred).response)
    return responses[r]


while True:
    text = [input()]
    if text[0].lower() in ['exit']:
        break
    df_input = pd.DataFrame(text,columns=['question'])
    no_stop = []
    for entry in df_input['question']:
        tokens = entry.split()
        punc_remove = re.compile('[%s]' % re.escape(string.punctuation))
        tokens = [punc_remove.sub('',w) for w in tokens]
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
        tokens = [word.lower() for word in tokens if len(word) > 1]
        joblib.dump(tokens,'tokens.pkl')
        no_stop.append(' '.join(tokens))
    df_input['question'] = no_stop

    t = tokenizer_t
    entry = [df_input['question'][0]]
    encoded = t.texts_to_sequences(entry)
    padded = pad_sequences(encoded, maxlen=16, padding='post')
    encoded_input = padded
    pred = get_pred(model,encoded_input)
    pred = bot_precausion(df_input,pred)

    response = get_response(dfs,pred)
    print(response)



