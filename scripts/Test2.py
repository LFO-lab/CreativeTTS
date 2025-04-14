import torch
from TTS.api import TTS

# using the default version set in 🐸TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# using a specific version
# 👀 see the branch names for versions on https://huggingface.co/coqui/XTTS-v2/tree/main
# ❗some versions might be incompatible with the API
#tts = TTS("xtts_v2.0.2", gpu=True)

# getting the latest XTTS_v2
#tts = TTS("xtts", gpu=True)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="Bonjour, je m'appelle Mimi Allar",
                file_path="output.wav",
                speaker_wav=["./training/Datasets/MIMI_DATASET_22-16/0.wav", "./training/Datasets/MIMI_DATASET_22-16/1.wav", "./training/Datasets/MIMI_DATASET_22-16/2.wav", "./training/Datasets/MIMI_DATASET_22-16/3.wav", "./training/Datasets/MIMI_DATASET_22-16/4.wav", "./training/Datasets/MIMI_DATASET_22-16/5.wav", "./training/Datasets/MIMI_DATASET_22-16/6.wav"],
                language="fr")