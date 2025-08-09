import time
import tensorflow as tf
from tensorflow_tts.inference import AutoProcessor, TFAutoModel
import soundfile as sf

# Load processor and models (done once)
processor = AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
tacotron2 = TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")

text = "Hello, this is a test of TensorFlow Tee Tee eSS."

input_ids = processor.text_to_sequence(text)

start_time = time.time()

# Tacotron2 inference
mel_outputs, _, _, _ = tacotron2.inference(
    tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    tf.convert_to_tensor([len(input_ids)], dtype=tf.int32),
    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
)

# Vocoder inference
audio = mb_melgan.inference(mel_outputs)[0, :, 0]

end_time = time.time()
duration = end_time - start_time

print(f"Generated audio in {duration:.3f} seconds.")

sf.write("output.wav", audio.numpy(), 22050)
print("Audio saved as output.wav")
