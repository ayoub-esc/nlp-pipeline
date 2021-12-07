# app.py 

from flask import Flask, request, render_template, make_response
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
import os
from sklearn import preprocessing
from transformers import TFXLNetModel, XLNetTokenizer
from nltk.corpus import stopwords

nltk.download('stopwords')
stop = stopwords.words('english')


def xlnet_model():
    word_inputs = tf.keras.Input(shape=(128,), name='word_inputs', dtype='int32')
    xlnet = TFXLNetModel.from_pretrained('xlnet-base-cased')
    xlnet_encodings = xlnet(word_inputs)[0]
    doc_encoding = tf.squeeze(xlnet_encodings[:, -1:, :], axis=1)
    doc_encoding = tf.keras.layers.Dropout(.1)(doc_encoding)
    outputs = tf.keras.layers.Dense(10, activation='softmax', name='outputs')(doc_encoding)
    model = tf.keras.Model(inputs=[word_inputs], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss='categorical_crossentropy')

    return model


def tokenize(text, tokenizer, max_len=128):
    inps = [tokenizer.encode_plus(t, max_length=max_len, truncation=True, padding='max_length', add_special_tokens=True)
            for t in text]
    inp_tok = np.array([a['input_ids'] for a in inps])
    ids = np.array([a['attention_mask'] for a in inps])
    segments = np.array([a['token_type_ids'] for a in inps])
    return inp_tok, ids, segments


app = Flask(__name__)
UPLOAD_FOLDER = app.root_path + '/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DATA_PATH = app.root_path + '/data/'
MODEL_PATH = app.root_path + '/models/'
xlnet = xlnet_model()
xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
xlnet = xlnet_model()
xlnet.load_weights(MODEL_PATH + "xlnet_finetuned.h5")
encoder = preprocessing.LabelEncoder()
encoder.classes_ = np.load(MODEL_PATH + 'encoder.npy', allow_pickle=True)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            text = request.form['message']
            code = text_note(text)
            return render_template('index.html', code=code)
        else:
            file = request.files['file']
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            render_template('index.html')
            csv = file_notes(filename)
            return csv
    else:
        return render_template('index.html')


def file_notes(filename):
    df = pd.read_csv(UPLOAD_FOLDER + "/" + filename)
    codes = []
    for x in df:
        codes.append(text_note(x))
    codes_list = ",".join(codes)
    response = make_response(codes_list)
    cd = 'attachment; filename=ICD-10_Codes.csv'
    response.headers['Content-Disposition'] = cd
    response.mimetype = 'text/csv'
    return response


def text_note(text):
    text = text.replace(r'[^\w\s]+', '')
    words = text.split()
    text = ' '.join([string for string in words if string not in stop])
    inp_tok, ids, segments = tokenize([text], xlnet_tokenizer)
    preds = xlnet.predict(inp_tok, verbose=True)
    preds = np.argmax(preds, axis=1)
    code = encoder.inverse_transform(preds)
    return code[0]


if __name__ == '__main__':
    app.run()
