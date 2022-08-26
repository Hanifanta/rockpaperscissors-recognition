### Rock Paper Scissors Recognition

Link Dataset : https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors

## Import Library
```bash
import numpy as np 
import pandas as pd 
import os
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from zipfile import ZipFile as zip
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image
from google.colab import files
%matplotlib inline

for dirname, _, filenames in os.walk('/content'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        break
    break
```

## Extract ZIP Dataset
```bash
from zipfile import ZipFile as zip
import os

zip = ZipFile('/content/rockpaperscissors.zip')
zip.extractall()
```

## Make Folder for Train and Validation
```bash
BASE_DATASET_DIR = '/content/rockpaperscissors'
if not os.path.exists('/content/dicoding/validation'):
    os.mkdir('/content/dicoding/validation')
    os.mkdir('/content/dicoding/validation/rock')
    os.mkdir('/content/dicoding/validation/paper')
    os.mkdir('/content/dicoding/validation/scissors')
    
if not os.path.exists('/content/dicoding/testing'):
    os.mkdir('/content/dicoding/testing')

VALIDATION_DATASET_DIR = '/content/dicoding/validation/'
TESTING_DATASET_DIR = '/content/dicoding/testing/'
```

```bash
print(len(os.listdir("/content/dicoding/rock")))
print(len(os.listdir("/content/dicoding/scissors")))
print(len(os.listdir("/content/dicoding/paper")))
```

```bash
train_rock = os.path.join(BASE_DATASET_DIR, classes[0])
train_paper = os.path.join(BASE_DATASET_DIR, classes[1])
train_scissor = os.path.join(BASE_DATASET_DIR, classes[2])

valid_rock = os.path.join(VALIDATION_DATASET_DIR, classes[0])
valid_paper = os.path.join(VALIDATION_DATASET_DIR, classes[1])
valid_scissor = os.path.join(VALIDATION_DATASET_DIR, classes[2])
```

```bash
print("Training Dataset")

print(len(os.listdir(train_rock)))
print(len(os.listdir(train_paper)))
print(len(os.listdir(train_scissor)))

print("Validation dataset count")

print(len(os.listdir(valid_rock)))
print(len(os.listdir(valid_paper)))
print(len(os.listdir(valid_scissor)))

print("Testing dataset count")
print(len(os.listdir(TESTING_DATASET_DIR)))
```

## Make Validation_Split

```bash
train_datagen = ImageDataGenerator(validation_split=0.4,
                                   rescale=1/255,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   rotation_range=0.1,
                                   zoom_range=0.1,
                                   fill_mode="nearest",
                                   vertical_flip=True,
                                   horizontal_flip=True
                                  )

train_generator = train_datagen.flow_from_directory('/content/dicoding/rps-cv-images',
                                                    target_size=(150, 150),
                                                    batch_size=32,
                                                    subset='training')
valid_generator = train_datagen.flow_from_directory('/content/dicoding/rps-cv-images',
                                                    target_size=(150,150),
                                                    batch_size=32,
                                                    subset='validation')
```

## Make Model for Training
```bash
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])
```

## Make Callback
```bash
class Callbackkuh(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= 0.97):
            print("\nReached %2.2f%% accuracy, training has been stop" %(logs.get('accuracy')*100))
            self.model.stop_training = True

callbacks = Callbackkuh()
```

## Make Model Compile
```bash
model.compile(loss='categorical_crossentropy',
              optimizer=tf.optimizers.Adam(),
              metrics=['accuracy'])
```

## Training Model
```bash
historyModel = model.fit(
    train_generator,
    steps_per_epoch = 41,
    epochs = 20,
    validation_data = valid_generator,
    validation_steps = 27,
    verbose = 2,
    callbacks = [callbacks]
    )
```

## Make Plot for Accuracy Check
```bash
acc = historyModel.history['accuracy']
val_acc = historyModel.history['val_accuracy']
loss = historyModel.history['loss']
val_loss = historyModel.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()
```

## Make upload files for try model
```bash
uploaded = files.upload()

for fn in uploaded.keys():
  path = fn
  img = image.load_img(path, target_size=(150,150))
  
  imgplot = plt.imshow(img)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  images = np.vstack([x])
  
  classes = model.predict(images, batch_size=10)
  if classes[0][0] == 1:
    print('Paper')
  elif classes[0][1] == 1:
    print('Rock')
  else:
    print('Scissors')
```

