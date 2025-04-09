import os
import torch
from TTS.api import TTS
from SimpleOSC import SimpleOSC

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS
tts = TTS(model_name="tts_models/fr/css10/vits", progress_bar=False).to(device)

# Output directory
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Init OSC
osc = SimpleOSC(send_ip="127.0.0.1", send_port=9000, receive_port=8000)

# OSC handler
def handle_speak(address, *args):
    if not args:
        print("No text provided.")
        return

    text = str(args[0])
    print(f"🔊 Received text: '{text}'")

    import time
    import re
    base_name = re.sub(r'\W+', '_', text[:30])
    filename = f"{base_name}_{int(time.time())}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    abs_path = os.path.abspath(filepath)  # 🔹 Chemin absolu ici

    try:
        # Générer le fichier audio
        print(f"🎙️ Génération de l’audio : {abs_path}")
        tts.tts_to_file(text=text, file_path=abs_path)
        print(f"✅ Terminé : {abs_path}")

        # Envoi du message OSC avec le chemin absolu
        osc.send("/done", abs_path)
        print(f"📤 Message OSC envoyé à /done avec le chemin : {abs_path}")

    except Exception as e:
        print(f"❌ Erreur lors de la génération : {e}")
        osc.send("/error", str(e))

# Bind OSC handler
osc.on("/speak", handle_speak)

# Start server
print("🚀 Serveur OSC TTS en cours d’exécution. Envoyez vos messages à /speak.")
osc.start()

try:
    input("Appuyez sur Entrée pour arrêter le serveur...\n")
finally:
    osc.stop()
    print("👋 Serveur arrêté.")
