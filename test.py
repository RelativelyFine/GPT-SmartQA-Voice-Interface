from utils import play 
from simple import generate

import socks
import socket

socks.set_default_proxy(socks.SOCKS5, "localhost", 9150)
socket.socket = socks.socksocket

audio = generate (
  # api_key=eleven_labs,
  text=''
  voice="Bella",
  model="eleven_monolingual_v1",
)
play(audio)