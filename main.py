#! python3.10.9
import threading
import queue
import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch
import openai
import string
import nltk
import re 

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
from dotenv import load_dotenv
from utils import stream 
from simple import generate

try:
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    eleven_labs = os.getenv("ELEVENLABS_API_KEY")
except:
    print("Error: Could not load .env file.")
    exit(1)

class Transcriber:
    def __init__(self, args):
        self.args = args
        self.phrase_time = None
        self.last_sample = bytes()
        self.data_queue = Queue()
        self.recorder = self.setup_recorder()
        self.source = self.setup_microphone()
        self.audio_model = self.load_model()
        self.previousQueueEmpty = True
        self.transcription = ['']
        self.buffered_messages = ''
        self.api_key = "eleven_labs"
        self.audio_queue = queue.Queue()  # Create a Queue to hold audio
        self.waitingAudios = 0

    def setup_recorder(self):
        recorder = sr.Recognizer()
        recorder.energy_threshold = self.args.energy_threshold
        recorder.dynamic_energy_threshold = False
        return recorder

    def setup_microphone(self):
        if 'linux' in platform:
            mic_name = self.args.default_microphone
            if not mic_name or mic_name == 'list':
                self.print_microphone_devices()
                return
            else:
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        return sr.Microphone(sample_rate=48000, device_index=index)
        else:
            return sr.Microphone(sample_rate=48000)

    def load_model(self):
        model = self.args.model
        if self.args.model != "large" and not self.args.non_english:
            model = model + ".en"
        return whisper.load_model(model)

    def print_microphone_devices(self):
        print("Available microphone devices are: ")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"Microphone with name \"{name}\" found")

    def record_callback(self, _, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        self.data_queue.put(data)

    def start_transcribing(self):
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)
        self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=self.args.record_timeout)
        print("Model loaded.\n")
        self.transcribe_loop()

    def transcribe_loop(self):
        temp_file = NamedTemporaryFile().name
        print("Listening...")
        while True:
            try:
                now = datetime.utcnow()
                if not self.data_queue.empty() or (self.phrase_time and now - self.phrase_time > timedelta(seconds=self.args.phrase_timeout) and not self.previousQueueEmpty):
                    self.previousQueueEmpty = False
                    if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.args.phrase_timeout):
                        self.last_sample = bytes()
                        self.previousQueueEmpty = True
                    self.phrase_time = now

                    while not self.data_queue.empty():
                        data = self.data_queue.get()
                        self.last_sample += data

                    audio_data = sr.AudioData(self.last_sample, self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH)
                    wav_data = io.BytesIO(audio_data.get_wav_data())

                    with open(temp_file, 'w+b') as f:
                        f.write(wav_data.read())

                    result = self.audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
                    text = result['text'].strip()

                    if self.previousQueueEmpty:
                        self.transcription.append(text)
                        if any(phrase in self.transcription[-2].lower().translate(str.maketrans('', '', string.punctuation)) for phrase in ["hey april", "hi april", "yo april", "so april"]):
                            self.chat_with_gpt()
                    else:
                        self.transcription[-1] = text

                    os.system('cls' if os.name=='nt' else 'clear')
                    for line in self.transcription:
                        print(line)
                    print('', end='', flush=True)

                    sleep(0.2)
            except KeyboardInterrupt:
                break

        print("\n\nTranscription:")
        for line in self.transcription:
            print(line)

    def chat_with_gpt(self):
        print("triggered")
        """Chats with GPT-3.5-turbo using the last 20 phrases or less."""
        prompt = "\"" + "\". \"".join(self.transcription[-2:]).replace("april", "gpt") + "\""  # Get the last 20 transcriptions or less.
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {'role': 'user', 'content': prompt},
            ],
            temperature=0,
            stream=True
        )
        generate_thread = threading.Thread(target=self.generate_and_queue, args=(response,))
        stream_thread = threading.Thread(target=self.stream_from_queue)

        # Start the threads
        generate_thread.start()
        stream_thread.start()

        # Wait for both threads to finish
        generate_thread.join()
        stream_thread.join()

        # Append the entire chat to the transcription
        self.transcription.append(self.buffered_messages)

    def is_sentence(self, text):
        pattern = r'^[^.]*[^\.0-9][.!?]$'
        match = re.match(pattern, text)
        return match is not None

    def generate_and_queue(self, response):
        for chunk in response:
            chunk_message = chunk['choices'][0]['delta']
            if 'content' in chunk_message:
                self.buffered_messages += chunk_message['content']  # append the message
                sentences = nltk.sent_tokenize(self.buffered_messages)
                print(chunk_message['content'], end='')
                
                if self.waitingAudios > 0:
                    continue

                # Find the last complete sentence
                if len(sentences) >= 2 and self.is_sentence(sentences[-1]):
                    sentence = sentences.pop(0)  # remove and get the first sentence 
                    while sentences:
                        sentence += sentences.pop(0) # append the next sentence

                    self.buffered_messages = ' '.join(sentences)  # update the buffer with the remaining sentences
                    audio = self.generate(sentence)
                    self.waitingAudios += 1
                    self.audio_queue.put(audio)
                    print("\n!!Sentenced!!!\n")

        if self.buffered_messages:
            audio = self.generate(self.buffered_messages)
            self.waitingAudios += 1
            self.audio_queue.put(audio)
            print("\n!!!!!!!!!\n")
        self.buffered_messages = ''

        # Add a sentinel to signal we're done generating
        self.audio_queue.put(None)
        print("DONE!!!")

    def stream_from_queue(self):
        while True:
            audio = self.audio_queue.get()
            if audio is None:  # We're done streaming
                break
            self.waitingAudios -= 1
            stream(audio)
            self.audio_queue.task_done()
            

    def generate(self, text):
        audio = generate (
            api_key=eleven_labs,
            text=text,
            voice="Bella",
            model="eleven_monolingual_v1",
            stream=True
        )
        return audio

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true', help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=100, help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=1.6, help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=2.5, help="How much empty space between recordings before we consider it a new line in the transcription.", type=float)  
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse', help="Default microphone name for SpeechRecognition. Run this with 'list' to view available Microphones.", type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    transcriber = Transcriber(args)
    transcriber.start_transcribing()

if __name__ == "__main__":
    main()