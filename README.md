# Conversor de Temperaturas con Red Neuronal Feedforward

## Descripción

Este proyecto utiliza una red neuronal feedforward para convertir temperaturas de grados Celsius a grados Fahrenheit. El modelo se entrena utilizando un conjunto de datos históricos y permite realizar predicciones a través de una interfaz gráfica sencilla utilizando la biblioteca Flet.

## Lógica

La conversión se basa en la fórmula:
$$ F = C \times \frac{9}{5} + 32 $$

## Estructura del Proyecto

- `data/`: Contiene el archivo CSV con datos de temperatura.
- `src/`: Contiene scripts para crear y entrenar el modelo, así como para realizar predicciones.
  - `model.py`: Define la arquitectura de la red neuronal.
  - `train.py`: Entrena el modelo y guarda el modelo entrenado.
  - `predict.py`: Realiza predicciones a través de la consola.
  - `predict_flet.py`: Realiza predicciones a través de una interfaz gráfica utilizando Flet.
- `README.md`: Información sobre el proyecto.
- `requirements.txt`: Dependencias necesarias.
- `.gitignore`: Archivos a ignorar en Git.

## Uso

1. Clona el repositorio.
2. Crea un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Linux/MacOS
   venv\Scripts\activate     # En Windows
3. Instala las dependencias:
    ```bash
    pip install -r requirements.txt

4. Ejecuta el script de entrenamiento para crear y guardar el modelo:
    ```bash
    python src/train.py

5. Para hacer predicciones usando la consola, ejecuta:
    ```bash
    python src/predict.py

6. Para hacer predicciones usando la interfaz gráfica, ejecuta:
    ```bash
    python src/predict_flet.py