import pandas as pd
from model import create_model
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# Cargar los datos desde el CSV
data = pd.read_csv('data/temperaturas.csv')

# Dividir los datos en características (X) y etiquetas (y)
X = data['Celsius'].values.reshape(-1, 1)
y = data['Fahrenheit'].values

# Dividir en conjunto de entrenamiento y prueba (usando todos los datos)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = create_model()

# Guardar las pérdidas durante el entrenamiento
history = model.fit(X_train, y_train, epochs=1000, verbose=0)

# Evaluar el modelo
loss = model.evaluate(X_test, y_test)
print(f'Error cuadrático medio: {loss}')

# Realizar predicciones en el conjunto de prueba
predicciones = model.predict(X_test)

# Calcular el MAE
mae = mean_absolute_error(y_test, predicciones)
print(f'Error Absoluto Medio: {mae}')

# Calcular R²
r2 = r2_score(y_test, predicciones)
print(f'Coeficiente de determinación R²: {r2}')

# Graficar las pérdidas durante el entrenamiento
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Pérdida')
plt.title('Pérdida durante el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

# Graficar las predicciones frente a los valores reales
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='blue', label='Valores Reales')
plt.scatter(X_test, predicciones, color='red', label='Predicciones', alpha=0.5)
plt.xlabel('Temperatura en Celsius')
plt.ylabel('Temperatura en Fahrenheit')
plt.title('Predicciones vs Valores Reales')
plt.legend()

# Mostrar ambas gráficas
plt.tight_layout()
plt.show()

# Guardar el modelo entrenado
model.save('modelo_temperatura.h5')
print("Modelo guardado como 'modelo_temperatura.h5'")