import tensorflow as tf
from tensorflow_tts.inference import AutoProcessor, TFAutoModel
import soundfile as sf

# Load processor and models
processor = AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
tacotron2 = TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")

# Input text
input_text = "Hello, this is a test of TensorFlow TTS."

# Process input text to IDs
input_ids = processor.text_to_sequence(input_text)

print("Input IDs:", input_ids)
print("Input length:", len(input_ids))

mel_outputs, _, _, _ = tacotron2.inference(
    tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
    tf.convert_to_tensor([len(input_ids)], dtype=tf.int32),
    speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
)
print("Mel outputs shape:", mel_outputs.shape)

audio = mb_melgan.inference(mel_outputs)[0, :, 0]
print("Audio shape:", audio.shape)

sf.write("output.wav", audio.numpy(), 22050)
print("Audio saved successfully!")
