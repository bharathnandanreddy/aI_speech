

# --- Imports for speech recognition and TTS ---
import json
from pydoc import text
from google import genai
import sounddevice as sd
import numpy as np
import threading
import queue
from silero_vad import load_silero_vad, get_speech_timestamps
import speech_recognition as sr
import pyttsx3
import re
def get_menu(filename="menu.json"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            menu = json.load(f)
    except Exception as e:
        print(f"Error loading menu from {filename}: {e}\nUsing default menu.")
        menu = {
            "Coffee": 3.0,
            "Espresso": 2.5,
            "Cappuccino": 3.5,
            "Latte": 4.0,
            "Tea": 2.0,
            "Muffin": 2.5,
            "Croissant": 3.0
        }
    return menu

api_key = "AIzaSyAPJAQS6OF_ul42p28PK1Cr7QjexeWun6Y"
client = genai.Client(api_key=api_key)
menu = get_menu()
menu_str = "\n".join([f"{item}: ${price:.2f}" for item, price in menu.items()])

def extract_message_and_cart(text):
    # Find JSON block
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    cart = None
    if json_match:
        try:
            cart = json.loads(json_match.group(1))
        except Exception:
            cart = None
        # Remove JSON block from message
        message = text.replace(json_match.group(0), '').strip()
    else:
        message = text
    return message, cart

def speak(text):
    # --- TTS engine setup ---
    tts_engine = pyttsx3.init()
    tts_engine.say(text)
    tts_engine.runAndWait()
    tts_engine.stop()

# --- Speech recognition setup ---
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

def recognize_speech_from_mic():
    model_vad = load_silero_vad()
    recognizer = sr.Recognizer()
    print("Speak now...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', blocksize=CHUNK_SIZE, callback=audio_callback):
        while True:
            try:
                audio_chunk = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            import torch
            audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32)
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            speech_timestamps = get_speech_timestamps(audio_tensor, model_vad, sampling_rate=SAMPLE_RATE)
            if not speech_timestamps:
                audio_queue.task_done()
                continue
            audio_pcm = (audio_chunk * 32767).astype(np.int16)
            audio_bytes = audio_pcm.tobytes()
            audio_data = sr.AudioData(audio_bytes, SAMPLE_RATE, 2)
            try:
                text = recognizer.recognize_google(audio_data)
                print("You (recognized):", text)
                audio_queue.task_done()
                return text
            except sr.UnknownValueError:
                print("Could not understand audio.")
            except sr.RequestError as e:
                print(f"SpeechRecognition error: {e}")
            audio_queue.task_done()

def main():
    # Use a single 'contents' list with 'parts' and 'role' for Gemini SDK
    contents = [
            {
                "role": "user",
                "parts": [
                    {"text": (
                        f"New Session started! You are a friendly coffee shop barista for 'Central Perk'. Here is the menu:\n{menu_str}\n"
                        "Greet the new customer when they say hello and ask for their order. "
                        "When they order, confirm the item and quantity, and keep a running cart as a JSON object. "
                        "After each order update, reply with the updated cart in JSON format ```json (key: item, value: quantity)```, "
                        "and a reply message. Do not tell the customer the total until they are done. "
                        "When the customer is done, immediately summarize the order, return the total price, "
                        "and end the conversation by saying 'Thank you for your order!'. "
                        "Only offer items from the menu. Do not ask for more orders after the user is done."
                    )}
                ]
            }
        ]
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents
    )
    contents.append({
        "role": "model",
        "parts": [{"text": response.text}]
    })
    print("Barista:", response.text)
    speak(response.text)
    main_cart = {}
    while True:
        user_input = recognize_speech_from_mic()
        contents.append({
            "role": "user",
            "parts": [{"text": user_input}]
        })

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )
        contents.append({
            "role": "model",
            "parts": [{"text": response.text}]
        })

        message, cart = extract_message_and_cart(response.text)
        if cart:
            main_cart = cart
        print("Cart:", main_cart)
        print("Barista:", message)
        speak(message)
        if "Thank you for your order" in response.text:
            print("Session ended. Start the script again for a new customer.")
            break

if __name__ == "__main__":
    main()

