import numpy as np
import tensorflow as tf

# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model('modelo_temperatura.h5')

# Función para predecir temperaturas Fahrenheit a partir de Celsius
def predecir_fahrenheit(celsius):
    celsius_array = np.array(celsius).reshape(-1, 1)  # Asegurarse de que sea un array 2D
    fahrenheit_predicciones = model.predict(celsius_array)
    return fahrenheit_predicciones.flatten()  # Devolver como array unidimensional

# Ingreso de datos por consola
def ingresar_datos_consola():
    temperaturas_celsius = input("Ingrese las temperaturas en Celsius separadas por comas: ")
    temperaturas_celsius = [float(temp) for temp in temperaturas_celsius.split(',')]
    return temperaturas_celsius

# Ejecución principal
if __name__ == "__main__":
    nuevas_temperaturas_celsius = ingresar_datos_consola()
    predicciones = predecir_fahrenheit(nuevas_temperaturas_celsius)

    for celsius, fahrenheit in zip(nuevas_temperaturas_celsius, predicciones):
        print(f'{celsius} grados Celsius son {fahrenheit:.2f} grados Fahrenheit.')