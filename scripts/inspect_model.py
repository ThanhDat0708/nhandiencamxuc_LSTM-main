import tensorflow as tf
import numpy as np

model_path = '../model/speech_emotion_lstm_improved.keras'
print('Loading model from', model_path)
model = tf.keras.models.load_model(model_path, compile=False)
print('\nModel summary:')
model.summary()
last = model.layers[-1]
print('\nLast layer:', type(last), getattr(last, 'name', None))
try:
    print('Last layer config:')
    import json
    print(json.dumps(last.get_config(), indent=2))
except Exception as e:
    print('Could not get config:', e)
print('\nModel output shape:', model.output_shape)
# Generate random input of expected shape
x = np.random.randn(1,220,40).astype('float32')
print('\nRunning a random prediction...')
pred = model.predict(x)
print('Pred shape:', pred.shape)
print('Pred (raw):', pred)
print('Pred (as list):', [float(p) for p in pred[0]])
