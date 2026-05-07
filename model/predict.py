import numpy as np
import random
from utils.preprocess import preprocess_image

# Alphabet labels
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Demo prediction logic
# Replace with trained model later

def predict_sign(image):
    processed_image = preprocess_image(image)

    prediction = random.choice(labels)

    confidence = np.random.uniform(80, 99)

    return prediction, confidence
