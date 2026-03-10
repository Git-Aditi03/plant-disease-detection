import tensorflow as tf

# load keras model
model = tf.keras.models.load_model("plant_disease_model_final.keras")

# convert to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# save model
with open("plant_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted successfully!")
