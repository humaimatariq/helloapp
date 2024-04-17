from flask import Flask, request, render_template
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model(r"C:\Users\Umaima Tariq\Dropbox\My PC (DESKTOP-CVKKAE6)\Downloads\cavity_dec_new\cavity_dec\fine_tuned_model_cavity.h5")

# Function to perform prediction
def predict(image):
    # Preprocess the image
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (512, 512))
    image = np.expand_dims(image, axis=0)
    
    # Perform prediction
    prediction = model.predict(image)
    return str(prediction[0])  # Convert the prediction to a string

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    file = request.files['file']
    img = Image.open(file)
    prediction = predict(img)
    return prediction

if __name__== '_main_':
    app.run(debug=True)