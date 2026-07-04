from flask import Flask, render_template, request
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

MAX_LEN = 100  

model = tf.keras.models.load_model("bilstm_emotion_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def predict_emotion(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")
    prediction = model.predict(padded)
    class_index = np.argmax(prediction, axis=1)[0]
    emotion = label_encoder.inverse_transform([class_index])[0]
    confidence = np.max(prediction)
    return emotion, confidence

@app.route("/", methods=["GET", "POST"])
def home():
    emotion = None
    confidence = None

    if request.method == "POST":
        text = request.form["text"]
        emotion, confidence = predict_emotion(text)
        confidence = round(confidence * 100, 2)

    return render_template("index.html", emotion=emotion, confidence=confidence)

if __name__ == "__main__":
    print("Start running emotional detection......")
    app.run(debug=True)