import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# --- CONFIGURACIÓN DE RUTAS DINÁMICAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta del modelo (se asume que está en la misma carpeta que este script)
modelo_ruta = os.path.join(BASE_DIR, "color_classifier_model.h5")

# Directorio donde pondrás las imágenes para probar
# (Crea una carpeta llamada 'Predecir' junto al script)
directorio_imagenes = os.path.join(BASE_DIR, "Predecir")

# --- CARGA DEL MODELO ---
if os.path.exists(modelo_ruta):
    modelo = tf.keras.models.load_model(modelo_ruta)
else:
    print(f"Error: No se encontró el modelo en {modelo_ruta}")
    exit()

# Mapeo de índices de clase a nombres de colores
clases_a_colores = {
    0: "Azul", 1: "Blanco", 2: "Gris", 3: "Negro",
    4: "Plata", 5: "Rojo", 6: "Verde", 7: "Vino"
}

# --- PROCESAMIENTO DE IMAGEN ---
nombre_imagen = "imagen_prueba.jpg"  # Reemplaza con tu imagen de prueba
imagen_ruta = os.path.join(directorio_imagenes, nombre_imagen)

if os.path.exists(imagen_ruta):
    # Cargar y redimensionar
    img = image.load_img(imagen_ruta, target_size=(224, 224))
    img_array = image.img_to_array(img)
    
    # Normalización e Inferencia
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predicciones = modelo.predict(img_array)
    
    # Resultado
    clase_predicha = np.argmax(predicciones)
    color_predicho = clases_a_colores[clase_predicha]
    print(f"Color predicho: {color_predicho}")
else:
    print(f"Error: No se encontró la imagen en {imagen_ruta}")