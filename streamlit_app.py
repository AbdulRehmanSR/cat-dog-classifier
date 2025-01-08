import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from io import BytesIO

# Custom HTML for header
st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #f0f0f0;">
        <h1 style="color: #4CAF50;">Cat and Dog Classifier</h1>
        <p>Upload an image, and our AI model will predict whether it's a cat or a dog!</p>
    </div>
""", unsafe_allow_html=True)

image_size = (128, 128)

# Load the trained model
model = tf.keras.models.load_model('cat_and_dog.h5')

# Define the labels
labels = ['cats', 'dogs']

# Upload image section
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the uploaded image
    image = load_img(BytesIO(uploaded_file.read()), target_size=image_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.0  # Normalize the image

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make a prediction
    prediction = model.predict(image_array)
    predicted_label_index = np.argmax(prediction)
    predicted_label = labels[predicted_label_index]

    st.markdown(f"### Predicted Label: **{predicted_label}**")

    # Load test dataset for evaluation
    test_dataset_dir = 'dog_cat/'
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(test_dataset_dir, target_size=image_size, batch_size=1, class_mode='categorical', shuffle=False)

    # Predict classes for test images
    predictions = model.predict(test_generator)
    predicted_classes = predictions.argmax(axis=-1)
    true_classes = test_generator.classes

    # Calculate confusion matrix
    confusion = confusion_matrix(true_classes, predicted_classes)

    # Calculate F1 score
    f1 = f1_score(true_classes, predicted_classes, average='weighted')

    # Calculate accuracy
    accuracy = accuracy_score(true_classes, predicted_classes)

    st.markdown(f"### F1 Score: **{f1:.2f}**")
    st.markdown(f"### Accuracy: **{accuracy:.2f}**")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    st.pyplot(fig)
