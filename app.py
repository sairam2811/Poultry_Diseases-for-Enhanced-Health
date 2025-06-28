import os
import numpy as np
from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf

app = Flask(__name__)

# ✅ Load the ResNet50-trained model
model = tf.keras.models.load_model("best_model.h5")

# Class labels (in training order)
class_labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

# Folder for uploaded images
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    success = False
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # 👉 (Optional) Save, log or process the message here

        success = True
    return render_template('contact.html', success=success)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    predicted_class = None
    uploaded_image_path = None

    if request.method == 'POST':
        file = request.files['pc_image']
        if file:
            filename = file.filename
            uploaded_image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(uploaded_image_path)

            # ✅ Preprocess image for ResNet50
            img = load_img(uploaded_image_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # ✅ Predict
            prediction = model.predict(img_array)
            predicted_class = class_labels[np.argmax(prediction)]

    return render_template(
        'predict.html',
        predict=predicted_class,
        uploaded_image=uploaded_image_path if predicted_class else None
    )

if __name__ == '__main__':
    app.run(debug=True)
