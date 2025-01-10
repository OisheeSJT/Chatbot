import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import random
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return bag

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    results_index = np.argmax(res)
    tag = classes[results_index]
    return tag

def get_response(ints, intents_json):
    try:
        result = ""
        for i in intents_json['intents']:
            if i['tag'] == ints:
                result = random.choice(i['responses'])
                break
        return result
    except IndexError:
        return "I don't understand..."

while True:
    message = input("You:")
    ints = predict_class(message, model)
    res = get_response(ints, intents)
    print("Bot:", res)
