import os
import re
import time
import queue
import threading
import torch
from TTS.api import TTS
from SimpleOSC import SimpleOSC

# ==== CONFIG ====
OUTPUT_DIR = "./output"
MODEL_NAME = "tts_models/fr/css10/vits"
OSC_RECEIVE_PORT = 8000
OSC_SEND_PORT = 9000
OSC_IP = "127.0.0.1"

# ==== INIT ====
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name=MODEL_NAME, progress_bar=False).to(device)
osc = SimpleOSC(send_ip=OSC_IP, send_port=OSC_SEND_PORT, receive_port=OSC_RECEIVE_PORT)

# ==== FILE QUEUE ====
tts_queue = queue.Queue()

# ==== WORKER THREAD ====
def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break  # Signal pour terminer le thread proprement

        try:
            base_name = re.sub(r'\W+', '_', text[:30])
            filename = f"{base_name}_{int(time.time())}.wav"
            filepath = os.path.join(OUTPUT_DIR, filename)
            abs_path = os.path.abspath(filepath)

            print(f"🎙️ Inférence : {text}")
            tts.tts_to_file(text=text, file_path=abs_path)
            print(f"✅ Terminé : {abs_path}")

            osc.send("/done", abs_path)
            print(f"📤 /done envoyé avec : {abs_path}")

        except Exception as e:
            print(f"❌ Erreur durant l'inférence : {e}")
            osc.send("/error", str(e))

        finally:
            tts_queue.task_done()

# ==== OSC HANDLER ====
def handle_speak(address, *args):
    if not args:
        print("❌ Aucun texte reçu.")
        osc.send("/error", "Usage: /speak <text>")
        return

    text = str(args[0])
    print(f"📩 Reçu pour inférence : {text}")
    tts_queue.put(text)

# ==== SETUP ====
osc.on("/speak", handle_speak)
osc.start()

worker_thread = threading.Thread(target=tts_worker, daemon=True)
worker_thread.start()

print("🚀 Serveur OSC prêt (inférence en file unique). Usage: /speak <texte>")

# ==== STOP ====
try:
    input("Appuyez sur Entrée pour arrêter...\n")
finally:
    tts_queue.put(None)  # pour stopper le thread proprement
    worker_thread.join()
    osc.stop()
    print("👋 Serveur arrêté.")
