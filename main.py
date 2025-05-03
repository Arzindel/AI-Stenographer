# main.py

import os
import threading
import sounddevice as sd
import numpy as np
import queue
import requests
import sseclient
import pyautogui
import json
import time
import logging
from pynput import keyboard
from faster_whisper import WhisperModel
from config.settings import AI_INSTRUCTIONS
import tkinter as tk
from tkinter import ttk
import pygetwindow  # Import for handling active window
# import pyperclip    # Not used since we're avoiding clipboard operations

# Set up logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(filename='logs/app.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Debug flag
DEBUG = True  # Set to False to disable debug prints

# Initialize global variables
audio_queue = queue.Queue()
segment_queue = queue.Queue()
transcription_queue = queue.Queue()
response_queue = queue.Queue()
segment_id = 0
recording = False
pausing = False
segment_lock = threading.Lock()
selected_device_index = None

# Conversation history
conversation_history = []

# Load the Whisper model
model_size = "base"
model = WhisperModel(model_size, device="cpu", compute_type="int8")  # Using CPU to avoid CUDA issues

# Function to handle audio recording
def record_audio(current_segment_id):
    global recording, pausing, selected_device_index
    samplerate = 16000  # 16kHz sampling rate
    channels = 1
    try:
        with sd.InputStream(samplerate=samplerate, channels=channels, dtype='int16', device=selected_device_index) as stream:
            print("Recording started")
            audio_buffer = []
            while recording and not pausing:
                data, overflowed = stream.read(1024)
                audio_buffer.append(data.copy())
            # After recording stops or pauses, add the last segment
            if audio_buffer:
                # Concatenate the audio data
                segment_data = np.concatenate(audio_buffer, axis=0)
                # Flatten the array to 1D
                segment_data = segment_data.flatten()
                # Convert to float32 and normalize to range [-1.0, 1.0]
                segment_data = segment_data.astype(np.float32) / 32768.0
                segment_queue.put((current_segment_id, segment_data))
                print(f"Segment {current_segment_id} queued for processing")
    except Exception as e:
        logging.error(f"Error in record_audio: {e}")
        print(f"Error in record_audio: {e}")

# Function to transcribe audio segments
def transcribe_segments():
    global transcription_queue
    while True:
        seg_id, segment = segment_queue.get()
        try:
            print(f"Transcribing segment {seg_id}")
            # Transcribe the segment
            segments, info = model.transcribe(segment, language="en")
            # Concatenate the text from all segments
            transcription = ''.join([segment.text for segment in segments])
            # For debugging purposes, print the transcription
            if DEBUG:
                print(f"Transcription for segment {seg_id}: {transcription}")
            transcription_queue.put((seg_id, transcription))
            print(f"Segment {seg_id} transcription complete")
        except Exception as e:
            logging.error(f"Error in transcribe_segments: {e}")
            print(f"Error in transcribe_segments: {e}")

# Function to get the active window title
def get_active_window_title():
    active_window = pygetwindow.getActiveWindow()
    if active_window:
        return active_window.title
    return ''

# Function to type text quickly
def type_text_quickly(text):
    window_title = get_active_window_title().lower()
    # Determine the key combination based on the application
    if any(app in window_title for app in ['slack', 'discord', 'teams', 'chat', 'skype', 'zoom']):
        newline_keys = ['shift', 'enter']  # Use Shift + Enter for new lines in chat applications
    else:
        newline_keys = ['enter']  # Use Enter key in other applications

    # Split text into lines
    lines = text.split('\n')
    for i, line in enumerate(lines):
        pyautogui.write(line, interval=0)  # Type the line without delay
        if i < len(lines) - 1:
            # Simulate the key combination for new line using hotkey
            pyautogui.hotkey(*newline_keys)

# Function to send transcription to AI and type the response
def send_to_ai():
    global conversation_history, last_processed_id
    last_processed_id = -1

    while True:
        seg_id, transcription = transcription_queue.get()
        # Ensure segments are processed in order
        if seg_id != last_processed_id + 1:
            # Put it back in the queue and wait
            transcription_queue.put((seg_id, transcription))
            time.sleep(0.1)
            continue
        try:
            print(f"Sending segment {seg_id} to AI API")
            # Add the user's transcription to the conversation history
            conversation_history.append({"role": "user", "content": transcription})

            # Prepare the data payload
            data = {
                "mode": "instruct",
                "stream": True,
                "messages": conversation_history
            }

            # API call
            url = "http://127.0.0.1:5000/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
            client = sseclient.SSEClient(stream_response)

            # Type the response into the focused application using buffered chunks
            assistant_message = ''
            buffer = ''
            buffer_lock = threading.Lock()
            typing_thread = None

            def type_buffered_text():
                nonlocal buffer
                while True:
                    time.sleep(0.1)  # Adjust the delay as needed
                    with buffer_lock:
                        if buffer:
                            text_to_type = buffer
                            buffer = ''
                        else:
                            break
                    type_text_quickly(text_to_type)

            for event in client.events():
                if event.data == '[DONE]':
                    break
                try:
                    payload = json.loads(event.data)
                    if 'choices' in payload and len(payload['choices']) > 0:
                        delta = payload['choices'][0]['delta']
                        if 'content' in delta:
                            chunk = delta['content']
                            assistant_message += chunk
                            with buffer_lock:
                                buffer += chunk
                            if typing_thread is None or not typing_thread.is_alive():
                                typing_thread = threading.Thread(target=type_buffered_text)
                                typing_thread.start()
                except Exception as e:
                    print(f"Error parsing event data: {e}")
                    print(f"Event data that caused error: {event.data}")
            # Wait for the typing thread to finish
            if typing_thread is not None:
                typing_thread.join()
            # Add AI response to conversation history
            conversation_history.append({"role": "assistant", "content": assistant_message})
            print(f"Segment {seg_id} AI response complete")
            last_processed_id = seg_id
        except Exception as e:
            logging.error(f"Error in send_to_ai: {e}")
            print(f"Error in send_to_ai: {e}")

# Function to handle keyboard events
class KeyboardListener:
    def __init__(self):
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.press_count = 0
        self.timer = None
        self.double_press_timeout = 0.5  # 500 milliseconds

    def start(self):
        self.listener.start()

    def on_press(self, key):
        try:
            if key == keyboard.Key.insert:
                self.press_count += 1
                # Reset the timer
                if self.timer:
                    self.timer.cancel()
                self.timer = threading.Timer(self.double_press_timeout, self.check_press_count)
                self.timer.start()
        except Exception as e:
            logging.error(f"Error in on_press: {e}")
            print(f"Error in on_press: {e}")

    def check_press_count(self):
        global recording, pausing, segment_id, conversation_history, last_processed_id
        press_count = self.press_count

        if press_count == 1:
            # Single press action
            print("Insert key single press detected")
            if recording and not pausing:
                with segment_lock:
                    # Finish current segment
                    recording = False
                    print(f"Segment {segment_id} finished recording")
                # Start transcribing the current segment
                recording = True
                threading.Thread(target=record_audio, args=(segment_id,)).start()
                segment_id += 1
                print(f"Segment {segment_id} started recording")
        elif press_count == 2:
            # Double press action
            print("Insert key double press detected")
            if not recording:
                # Start recording
                recording = True
                pausing = False
                threading.Thread(target=record_audio, args=(segment_id,)).start()
                print("Recording started by double press")
            else:
                if pausing:
                    # Resume recording
                    pausing = False
                    recording = True
                    threading.Thread(target=record_audio, args=(segment_id,)).start()
                    print("Recording resumed by double press")
                else:
                    # Pause recording
                    pausing = True
                    recording = False
                    print(f"Recording paused, segment {segment_id} ended")
                    # Ensure last segment is queued
                    with segment_lock:
                        if recording:
                            recording = False
                        segment_id += 1
        elif press_count == 3:
            # Triple press action
            print("Insert key triple press detected")
            # Reset conversation history and segment ID
            conversation_history = []
            segment_id = 0
            last_processed_id = -1  # Reset last_processed_id
            # Clear queues
            with segment_lock:
                while not segment_queue.empty():
                    segment_queue.get()
                while not transcription_queue.empty():
                    transcription_queue.get()
            print("Conversation history, segment ID, and last_processed_id reset. Starting new session.")
        # Reset the press count
        self.press_count = 0

# Function to update selected device index
def update_device_index(event):
    global selected_device_index
    selected_device_index = int(device_list.get().split(':')[0])
    print(f"Selected device index: {selected_device_index}")

# Start threads
transcribe_thread = threading.Thread(target=transcribe_segments, daemon=True)
transcribe_thread.start()

ai_thread = threading.Thread(target=send_to_ai, daemon=True)
ai_thread.start()

keyboard_listener = KeyboardListener()
keyboard_listener.start()

# Create the application window
root = tk.Tk()
root.title("Speech-to-Text Application")
root.geometry("500x200")

# Create device selection UI
device_label = ttk.Label(root, text="Select Audio Input Device:")
device_label.pack(side=tk.TOP, padx=5, pady=5)

# Get list of input devices
input_devices = []
for idx, device in enumerate(sd.query_devices()):
    if device['max_input_channels'] > 0:
        input_devices.append(f"{idx}: {device['name']}")

device_list = ttk.Combobox(root, values=input_devices, width=60)
device_list.pack(side=tk.TOP, padx=5, pady=5)
device_list.bind("<<ComboboxSelected>>", update_device_index)

# Set default device
default_device_index = sd.default.device[0]  # Default input device
if default_device_index is not None:
    for idx, device in enumerate(input_devices):
        if int(device.split(':')[0]) == default_device_index:
            device_list.current(idx)
            selected_device_index = default_device_index
            break

label = tk.Label(root, text="Press the Insert key to control recording")
label.pack(side=tk.TOP, padx=5, pady=10)

# Start the Tkinter event loop
root.mainloop()
