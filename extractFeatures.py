from __future__ import print_function  # In python 2.7
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import logging
import sys
import os
import sys
import random
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Audio
from keras.models import Sequential, model_from_json
import pickle
from keras_preprocessing.sequence import pad_sequences
import warnings
from deepface import DeepFace
import cv2


app = Flask(__name__)

# sound
json_file_sound = open("Emotion_Model_conv1d_gender_93.json", 'r')
loaded_model_json_sound = json_file_sound.read()
json_file_sound.close()
loaded_model_sound = model_from_json(loaded_model_json_sound)

# load weights into new model
loaded_model_sound.load_weights("Emotion_Model_conv1d_gender_93.h5")
loaded_model_sound.compile(optimizer='RMSprop',
                           loss='categorical_crossentropy', metrics=['accuracy'])

# sample_rate = 22050
# this for text model
################################
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
json_file_text = open('Emotion_Model_bidirectional_text_93.json', 'r')
loaded_model_json_text = json_file_text.read()
json_file_text.close()
loaded_model_text = model_from_json(loaded_model_json_text)

# load weights into new model
loaded_model_text.load_weights("Emotion_Model_bidirectional_text_93.h5")
print("Loaded model from disk")


def extract_features(data):

    result = np.array([])

    mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=58)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    result = np.array(mfccs_processed)

    return result


def noise(data):
    noise_amp = 0.04*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data


def stretch(data, rate=0.70):
    return librosa.effects.time_stretch(data, rate=rate)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5)*1000)
    return np.roll(data, shift_range)


def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)


def higher_speed(data, speed_factor=1.25):
    return librosa.effects.time_stretch(data, rate=speed_factor)


def lower_speed(data, speed_factor=0.75):
    return librosa.effects.time_stretch(data, rate=speed_factor)


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=3, offset=0.5)

    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)

    return result


@app.route('/predictText', methods=['POST'])
def predictText():
    data = request.get_json()  # Parse JSON data from request body

    dictionary = {'joy': 0, 'anger': 1, 'love': 2,
                  'sadness': 3, 'fear': 4, 'surprise': 5}
    sentence = data["input"]
    sentence_lst = []
    sentence_lst.append(sentence)
    sentence_seq = tokenizer.texts_to_sequences(sentence_lst)
    sentence_padded = pad_sequences(sentence_seq, maxlen=80, padding='post')
    ans = np.argmax(loaded_model_text.predict(sentence_padded), axis=1)
    emotion_text = "not clear"
    for key, val in dictionary.items():
        if (val == ans):
            print("The emotion predicted is", key)
            emotion_text = key

    # Return the predicted label as JSON

    return jsonify({"emotion": emotion_text})


@app.route('/predictSound', methods=['POST'])
def predictSound():
    # Get the file from the request
    file = request.files['audio']

    # Save the file to a temporary directory
    tmp_dir = 'tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, file.filename)
    file.save(tmp_path)

    # Remove the temporary audio file
    # os.remove(tmp_path)
    category_dict = {7: 'male_angry', 8:  'male_disgust', 9: 'male_fear', 10: 'male_happy', 11: 'male_neutral', 12: 'male_sad', 13: 'male_surprise', 4: 'female_neutral',
                     3: 'female_happy', 5: 'female_sad', 0: 'female_angry', 2: 'female_fear',
                     1: 'female_disgust', 6: 'female_surprise', 16: "test"}

    feature = get_features(tmp_path)
    #
    feature = feature.reshape(1, feature.shape[0])
    print("feature")
    print(feature)
    X_expanded = np.expand_dims(feature, axis=2)
    # print(len(X_expanded), file=sys.stderr)

    pred = loaded_model_sound.predict(X_expanded)
    print("pred", file=sys.stderr)
    print(pred, file=sys.stderr)
    print(len(pred), file=sys.stderr)

    index = np.argmax(pred)
    print("index", file=sys.stderr)
    print(index, file=sys.stderr)

    label = category_dict[index]

    # Return the predicted label as JSON
    return jsonify({"emotion": label})


@app.route('/predictImage', methods=['POST'])
def predictImage():
    # Get the file from the request
    file = request.files['image']

    # Read the image from the file object
    image_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Perform your image processing or prediction on the image
    # For example:
    # Predictions = DeepFace.analyze(img)
    Predictions = DeepFace.analyze(
        img, actions=['age', 'gender', 'race', 'emotion'])
    label = Predictions[0]['dominant_emotion']
    gender = Predictions[0]['dominant_gender']
    # print(Predictions[0])

    # Return the predicted label and gender as JSON
    return jsonify({"emotion": label, "gender": gender})


if __name__ == '__main__':
    app.run(port=3000, debug=True)
