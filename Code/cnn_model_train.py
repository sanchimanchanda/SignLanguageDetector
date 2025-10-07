# # import numpy as np
# # import pickle
# # import cv2, os
# # from glob import glob
# # from keras import optimizers
# # from keras.models import Sequential
# # from keras.layers import Dense
# # from keras.layers import Dropout
# # from keras.layers import Flatten
# # from keras.layers.convolutional import Conv2D
# # from keras.layers.convolutional import MaxPooling2D
# # from keras.utils import np_utils
# # from keras.callbacks import ModelCheckpoint
# # from keras import backend as K
# # K.set_image_dim_ordering('tf')

# # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# # def get_image_size():
# # 	img = cv2.imread('gestures/1/100.jpg', 0)
# # 	return img.shape

# # def get_num_of_classes():
# # 	return len(glob('gestures/*'))

# # image_x, image_y = get_image_size()

# # def cnn_model():
# # 	num_of_classes = get_num_of_classes()
# # 	model = Sequential()
# # 	model.add(Conv2D(16, (2,2), input_shape=(image_x, image_y, 1), activation='relu'))
# # 	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
# # 	model.add(Conv2D(32, (3,3), activation='relu'))
# # 	model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
# # 	model.add(Conv2D(64, (5,5), activation='relu'))
# # 	model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
# # 	model.add(Flatten())
# # 	model.add(Dense(128, activation='relu'))
# # 	model.add(Dropout(0.2))
# # 	model.add(Dense(num_of_classes, activation='softmax'))
# # 	sgd = optimizers.SGD(lr=1e-2)
# # 	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# # 	filepath="cnn_model_keras2.h5"
# # 	checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# # 	callbacks_list = [checkpoint1]
# # 	#from keras.utils import plot_model
# # 	#plot_model(model, to_file='model.png', show_shapes=True)
# # 	return model, callbacks_list

# # def train():
# # 	with open("train_images", "rb") as f:
# # 		train_images = np.array(pickle.load(f))
# # 	with open("train_labels", "rb") as f:
# # 		train_labels = np.array(pickle.load(f), dtype=np.int32)

# # 	with open("val_images", "rb") as f:
# # 		val_images = np.array(pickle.load(f))
# # 	with open("val_labels", "rb") as f:
# # 		val_labels = np.array(pickle.load(f), dtype=np.int32)

# # 	train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
# # 	val_images = np.reshape(val_images, (val_images.shape[0], image_x, image_y, 1))
# # 	train_labels = np_utils.to_categorical(train_labels)
# # 	val_labels = np_utils.to_categorical(val_labels)

# # 	print(val_labels.shape)

# # 	model, callbacks_list = cnn_model()
# # 	model.summary()
# # 	model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=15, batch_size=500, callbacks=callbacks_list)
# # 	scores = model.evaluate(val_images, val_labels, verbose=0)
# # 	print("CNN Error: %.2f%%" % (100-scores[1]*100))
# # 	#model.save('cnn_model_keras2.h5')

# # train()
# # K.clear_session();

# import numpy as np
# import pickle
# import cv2
# import os
# from glob import glob
# from tensorflow.keras import optimizers
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras import backend as K

# # Suppress TensorFlow logs
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# def get_image_size():
#     """Return the size of any gesture image (height, width)."""
#     img = cv2.imread('gestures/1/100.jpg', cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise FileNotFoundError("Sample image not found in gestures/1/. Make sure gesture images exist.")
#     return img.shape

# def get_num_of_classes():
#     """Count how many gesture folders exist."""
#     return len(glob('gestures/*'))

# image_y, image_x = get_image_size()  # shape returns (height, width)

# def cnn_model():
#     """Define CNN architecture."""
#     num_of_classes = get_num_of_classes()
#     model = Sequential([
#         Conv2D(16, (2, 2), activation='relu', input_shape=(image_y, image_x, 1)),
#         MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),

#         Conv2D(32, (3, 3), activation='relu'),
#         MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'),

#         Conv2D(64, (5, 5), activation='relu'),
#         MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'),

#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.2),
#         Dense(num_of_classes, activation='softmax')
#     ])

#     # Updated optimizer argument for TF2.x
#     sgd = optimizers.SGD(learning_rate=1e-2)
#     model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#     filepath = "cnn_model_keras2.h5"
#     checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
#     callbacks_list = [checkpoint]

#     return model, callbacks_list

# def train():
#     """Load pickled datasets, train CNN, and save the model."""
#     with open("train_images", "rb") as f:
#         train_images = np.array(pickle.load(f))
#     with open("train_labels", "rb") as f:
#         train_labels = np.array(pickle.load(f), dtype=np.int32)

#     with open("val_images", "rb") as f:
#         val_images = np.array(pickle.load(f))
#     with open("val_labels", "rb") as f:
#         val_labels = np.array(pickle.load(f), dtype=np.int32)

#     # Reshape images for Keras input
#     train_images = train_images.reshape(train_images.shape[0], image_y, image_x, 1)
#     val_images = val_images.reshape(val_images.shape[0], image_y, image_x, 1)

#     # One-hot encode labels
#     train_labels = to_categorical(train_labels)
#     val_labels = to_categorical(val_labels)

#     print("Validation labels shape:", val_labels.shape)

#     model, callbacks_list = cnn_model()
#     model.summary()

#     model.fit(
#         train_images, train_labels,
#         validation_data=(val_images, val_labels),
#         epochs=15,
#         batch_size=500,
#         callbacks=callbacks_list
#     )

#     scores = model.evaluate(val_images, val_labels, verbose=0)
#     print("✅ CNN Validation Accuracy: %.2f%%" % (scores[1] * 100))
#     print("❌ CNN Error: %.2f%%" % (100 - scores[1] * 100))

#     # Optionally save again to ensure latest weights stored
#     model.save('cnn_model_keras2_final.h5')
#     K.clear_session()

# if __name__ == "__main__":
#     train()

import numpy as np
import pickle
import cv2
import os
from glob import glob
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ----------------------- Helper Functions -----------------------

def get_image_size():
    """Return the size of any gesture image (height, width)."""
    # Try first image in first numeric gesture folder
    gesture_folders = sorted([g for g in os.listdir('gestures') if g.isdigit()], key=int)
    if not gesture_folders:
        raise FileNotFoundError("No gesture folders found in 'gestures/'")
    
    first_folder = f"gestures/{gesture_folders[0]}"
    first_images = os.listdir(first_folder)
    if not first_images:
        raise FileNotFoundError(f"No images found in folder {first_folder}")
    
    img = cv2.imread(os.path.join(first_folder, first_images[0]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Could not read sample image for shape")
    return img.shape

def get_gesture_mapping():
    """Return dictionary mapping folder number to 0-based class index."""
    gesture_folders = sorted([g for g in os.listdir('gestures') if g.isdigit()], key=int)
    folder_to_index = {int(f): i for i, f in enumerate(gesture_folders)}
    return folder_to_index

# ----------------------- CNN Model -----------------------

def cnn_model(image_y, image_x, num_classes):
    """Define CNN architecture."""
    model = Sequential([
        Conv2D(16, (2, 2), activation='relu', input_shape=(image_y, image_x, 1)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'),

        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    sgd = optimizers.SGD(learning_rate=1e-2)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    filepath = "cnn_model_keras2.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list

# ----------------------- Training -----------------------

def train():
    image_y, image_x = get_image_size()
    folder_to_index = get_gesture_mapping()
    num_classes = len(folder_to_index)

    # Load pickled datasets
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    with open("val_images", "rb") as f:
        val_images = np.array(pickle.load(f))
    with open("val_labels", "rb") as f:
        val_labels = np.array(pickle.load(f), dtype=np.int32)

    # Map labels to 0-based indices
    train_labels = np.array([folder_to_index[label] for label in train_labels])
    val_labels = np.array([folder_to_index[label] for label in val_labels])

    # Reshape images for Keras input
    train_images = train_images.reshape(train_images.shape[0], image_y, image_x, 1)
    val_images = val_images.reshape(val_images.shape[0], image_y, image_x, 1)

    # One-hot encode labels
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    val_labels = to_categorical(val_labels, num_classes=num_classes)

    print("Validation labels shape:", val_labels.shape)

    model, callbacks_list = cnn_model(image_y, image_x, num_classes)
    model.summary()

    # Train model
    model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=15,
        batch_size=64,
        callbacks=callbacks_list
    )

    scores = model.evaluate(val_images, val_labels, verbose=0)
    print("✅ CNN Validation Accuracy: %.2f%%" % (scores[1] * 100))
    print("❌ CNN Error: %.2f%%" % (100 - scores[1] * 100))

    # Save final model
    model.save('cnn_model_keras2_final.h5')
    K.clear_session()

# ----------------------- Main -----------------------

if __name__ == "__main__":
    train()
