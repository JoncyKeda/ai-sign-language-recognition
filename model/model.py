from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

def build_model():
    model = Sequential()

    # First CNN Layer
    model.add(
        Conv2D(
            32,
            (3,3),
            activation='relu',
            input_shape=(64,64,3)
        )
    )

    model.add(MaxPooling2D(pool_size=(2,2)))

    # Second CNN Layer
    model.add(
        Conv2D(
            64,
            (3,3),
            activation='relu'
        )
    )

    model.add(MaxPooling2D(pool_size=(2,2)))

    # Flatten
    model.add(Flatten())

    # Dense Layer
    model.add(Dense(128, activation='relu'))

    # Output Layer (26 alphabets)
    model.add(Dense(26, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
