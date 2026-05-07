import numpy as np

def preprocess_image(image):
    # Resize image
    image = image.resize((64,64))

    # Convert to numpy array
    img = np.array(image)

    # Normalize pixels
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img
