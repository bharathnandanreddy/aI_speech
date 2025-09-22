

from transformers import AutoProcessor, SeamlessM4Tv2Model
import torch
import sounddevice as sd
import numpy as np

import threading
import queue
from silero_vad import load_silero_vad, get_speech_timestamps

# Load model + processor
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")

SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # seconds
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
CHANNELS = 1

audio_queue = queue.Queue()
stop_event = threading.Event()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    # Flatten and copy to avoid referencing the same memory
    audio_chunk = np.copy(indata[:, 0])
    audio_queue.put(audio_chunk)

def transcribe_worker():
    print("Transcription worker started.")
    model_vad = load_silero_vad()
    while not stop_event.is_set():
        try:
            audio_chunk = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        # Silero expects torch tensor, 16kHz, mono, float32
        audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        speech_timestamps = get_speech_timestamps(audio_tensor, model_vad, sampling_rate=SAMPLE_RATE)
        if not speech_timestamps:
            audio_queue.task_done()
            continue
        inputs = processor(audio=audio_tensor, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        output_tokens = model.generate(**inputs, tgt_lang="eng", generate_speech=False)
        translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        print("Transcribed text:", translated_text_from_audio)
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
