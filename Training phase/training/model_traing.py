import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.utils import image_dataset_from_directory
from keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

image_size = (180, 180)
batch_size = 16
epochs = 100
patience = 6

dataset_path = "D:\\Graduation project\\Graduation Part1\\dataset\\training_data"
train_set = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset='training',
    seed=123,
    shuffle=True,
    image_size=image_size,
    batch_size=batch_size
)
valid_set = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset='validation',
    seed=123,
    shuffle=True,
    image_size=image_size,
    batch_size=batch_size
)

labels = train_set.class_names
# ['non_smoking', 'smoking']
# https://stackoverflow.com/questions/56613155/tensorflow-tf-data-autotune
# Performance optimization for Data Loading to CNN
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_set = train_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
valid_set = valid_set.cache().prefetch(buffer_size=AUTOTUNE)
# Data Generation
data_augmentation = Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip(
        'horizontal', input_shape=(180, 180, 3)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)
])
model = Sequential([
    data_augmentation,
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(1000, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(400, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax'),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
model.summary()
callbacks = [
    EarlyStopping(monitor='val_loss', patience=patience)
]
history = model.fit(
    train_set,
    validation_data=valid_set,
    callbacks=callbacks,
    epochs=epochs
)
loss, accuracy = model.evaluate(valid_set)
print('Test Loss : ', loss)
print('Test Accuracy : ', accuracy)

early_epoch = history.epoch[-1] + 1
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(early_epoch)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('ploy.jpg')
model.save('ModelV3')
