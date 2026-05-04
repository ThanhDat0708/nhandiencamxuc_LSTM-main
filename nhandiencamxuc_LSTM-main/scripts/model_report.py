import json
import tensorflow as tf

model_path = 'model/speech_emotion_lstm_improved.keras'
print('Loading model:', model_path)
model = tf.keras.models.load_model(model_path, compile=False)
print('\n=== MODEL SUMMARY ===')
model.summary()
print('\n=== OUTPUT SHAPE ===')
print(model.output_shape)
print('\n=== LAST LAYERS ===')
for layer in model.layers[-5:]:
    print(type(layer).__name__, layer.name)
    try:
        print(json.dumps(layer.get_config(), indent=2))
    except Exception as exc:
        print('config unavailable:', exc)
        print('---')
