import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 🔥 Patch layers
from tensorflow.keras.layers import Embedding, Dense, GRU

def patch_layer(layer_class):
    original_init = layer_class.__init__
    def new_init(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        original_init(self, *args, **kwargs)
    layer_class.__init__ = new_init

patch_layer(Embedding)
patch_layer(Dense)
patch_layer(GRU)

app = Flask(__name__)

with open('GRU_Tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = tf.keras.models.load_model('GRU_Spam_Detector.h5', compile=False)

MAX_LEN = 100

def predict_spam(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded)[0][0]
    return float(pred)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('message')

    pred = predict_spam(text)

    if pred > 0.5:
        result = "Spam"
        confidence = round(pred * 100, 2)
    else:
        result = "Not Spam"
        confidence = round((1 - pred) * 100, 2)

    return jsonify({
        "result": result,
        "confidence": confidence
    })

if __name__ == '__main__':
    app.run(debug=True)