import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- CONFIGURACIÓN DE RUTAS DINÁMICAS ---
# Esto detecta la carpeta donde se encuentra este archivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Definimos las rutas uniendo la base con el nombre de la carpeta
# Nota: Asegúrate de que tus carpetas se llamen exactamente así y estén junto al script
train_dir = os.path.join(BASE_DIR, "Entrenamiento")
val_dir = os.path.join(BASE_DIR, "Validacion")
model_save_path = os.path.join(BASE_DIR, "color_classifier_model.h5")

# --- GENERADOR DE DATOS ---
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Carga de datos
train_generator = datagen.flow_from_directory(
    train_dir, 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical'
)

val_generator = datagen.flow_from_directory(
    val_dir, 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical'
)

# --- DEFINICIÓN DEL MODELO (CNN) ---
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(8, activation="softmax")
])

# Compilación
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Entrenamiento
model.fit(train_generator, epochs=25, validation_data=val_generator)

# Guardado automático en la carpeta del proyecto
model.save(model_save_path)

# Evaluación
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {accuracy}")