# 🚗 Car Color Recognition AI (CNN)

> **Deep Learning & Computer Vision Project**
> Modelo diseñado para clasificar colores de vehículos mediante Redes Neuronales Convencionales (CNN), cubriendo el flujo completo desde la recolección de datos hasta la inferencia.

---

## 📖 Descripción del Proyecto
Este repositorio contiene el pipeline completo para la clasificación multiclase de tonalidades automotrices. El proyecto destaca por el uso de un **dataset propietario** generado mediante técnicas de recolección automatizada, procesado con arquitecturas de aprendizaje profundo.

## 📊 El Dataset y Preparación
* **Origen:** Datos obtenidos mediante técnicas avanzadas de **Web Scraping** de portales automotrices (código de recolección privado).
* **Estructura:** Aprendizaje supervisado con división de datos en carpetas de `Entrenamiento` y `Validación`.
* **Aumento de Datos (Data Augmentation):** Uso de `ImageDataGenerator` para aplicar rotaciones, zooms y volteos, asegurando que el modelo sea robusto ante variaciones del mundo real.

### Categorías de Color Soportadas:
* 🔵 **Azul** | ⚪ **Blanco** | 🔘 **Gris** | ⚫ **Negro**
* 🥈 **Plata** | 🔴 **Rojo** | 🟢 **Verde** | 🍷 **Vino**



## 🧠 Arquitectura del Modelo
El modelo se basa en una Red Neuronal Convencional (CNN) secuencial de alto rendimiento:
* **Capas Convolucionales:** 4 capas de `Conv2D` con filtros crecientes (32, 64, 128 y 256).
* **Pooling:** Capas de `MaxPooling2D` para reducir la dimensionalidad.
* **Capa Densa:** Una capa totalmente conectada de 128 neuronas con activación `ReLU`.
* **Salida:** Clasificación Multiclase mediante función `Softmax`.



## 🛠️ Stack Tecnológico
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

---

## 📂 Estructura del Repositorio
Para mantener el repositorio limpio y profesional, solo se incluyen los scripts fuente:

```text
.
├── train.py              # Script de entrenamiento y generación del modelo (.h5)
├── predict.py            # Script de inferencia y testeo individual
├── requirements.txt      # Dependencias del proyecto
├── /Entrenamiento        # (Usuario) Imágenes para entrenamiento
├── /Validacion           # (Usuario) Imágenes para validación
└── /Predecir             # (Usuario) Imágenes para realizar predicciones

## 📊 Ejemplo de Funcionamiento
Para validar el modelo, se realizó una prueba de inferencia con una imagen externa no vista durante el entrenamiento:

| Imagen de Entrada | Resultado de la Predicción |
| :---: | :---: |
| ![Auto de Prueba](![Camioneta](https://github.com/user-attachments/assets/aa21c8f3-9ff4-4692-b61f-dd60ad58bd4f)
) | ![Resultado Terminal](<img width="292" height="46" alt="Captura Terminal" src="https://github.com/user-attachments/assets/ccbda653-f330-404c-9722-a04d5f5276ff" />
) |

> **Nota:** El modelo identifica correctamente las características tonales y aplica la clasificación en menos de 1 segundo.
