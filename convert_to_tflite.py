import tensorflow as tf

keras_model = tf.keras.models.load_model("newest_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

# FP32 dönüşüm için optimizasyonu kapatabilir ya da açık bırakabilirsin
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]


tflite_fp32 = converter.convert()

with open("newest_model_fp32.tflite", "wb") as f:
    f.write(tflite_fp32)

print("Float32 model boyutu:", len(tflite_fp32) / 1024**2, "MB")
