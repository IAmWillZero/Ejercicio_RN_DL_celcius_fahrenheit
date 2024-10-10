import tensorflow as tf

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),  # Aumentar a 10 neuronas
        tf.keras.layers.Dense(1)  # Capa de salida
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model