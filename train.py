import tensorflow as tf
import datetime
from data_preparation import train_dataset, test_dataset, num_labels
from model import create_model

print("TensorFlow Version:", tf.__version__)

# Create model using a custom function that you've defined in the `model.py` script
model = create_model(num_labels)

# Setting up the optimizer, loss, and metrics
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.CategoricalAccuracy('accuracy')

# Compile the model with the specified optimizer, loss, and metrics
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Prepare datasets by adjusting batch size and prefetching
train_dataset = train_dataset.batch(16).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(16).prefetch(tf.data.AUTOTUNE)

# Define callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'model_best', save_best_only=True, verbose=1, save_format='tf')
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, verbose=1)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

callbacks_list = [checkpoint, reduce_lr, early_stopping, tensorboard]

# Fit the model
model.fit(
    train_dataset,
    epochs=3,
    validation_data=test_dataset,
    callbacks=callbacks_list,
    verbose=2
)
