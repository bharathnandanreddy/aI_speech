import sounddevice as sd
import numpy as np
import threading
import queue
from silero_vad import load_silero_vad, get_speech_timestamps
import speech_recognition as sr

SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # seconds
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
CHANNELS = 1

audio_queue = queue.Queue()
stop_event = threading.Event()

def audio_callback(indata, frames, time, status):
	if status:
		print(status)
	audio_chunk = np.copy(indata[:, 0])
	audio_queue.put(audio_chunk)

def transcribe_worker():
	print("Transcription worker started.")
	model_vad = load_silero_vad()
	recognizer = sr.Recognizer()
	while not stop_event.is_set():
		try:
			audio_chunk = audio_queue.get(timeout=0.5)
		except queue.Empty:
			continue
		# Silero expects torch tensor, 16kHz, mono, float32
		import torch
		audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32)
		if audio_tensor.ndim == 1:
			audio_tensor = audio_tensor.unsqueeze(0)
		speech_timestamps = get_speech_timestamps(audio_tensor, model_vad, sampling_rate=SAMPLE_RATE)
		if not speech_timestamps:
			audio_queue.task_done()
			continue
		# Convert numpy float32 audio to 16-bit PCM for SpeechRecognition
		audio_pcm = (audio_chunk * 32767).astype(np.int16)
		audio_bytes = audio_pcm.tobytes()
		audio_data = sr.AudioData(audio_bytes, SAMPLE_RATE, 2)
		try:
			text = recognizer.recognize_google(audio_data)
			print("Transcribed text:", text)
		except sr.UnknownValueError:
			print("Could not understand audio.")
		except sr.RequestError as e:
			print(f"SpeechRecognition error: {e}")
		audio_queue.task_done()

def main():
	print("Speak into the microphone. Press Ctrl+C to stop.")
	worker = threading.Thread(target=transcribe_worker, daemon=True)
	worker.start()
	try:
		with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', blocksize=CHUNK_SIZE, callback=audio_callback):
			while True:
				sd.sleep(1000)
	except KeyboardInterrupt:
		print("\nStopped.")
		stop_event.set()
		worker.join()

if __name__ == "__main__":
	main()
