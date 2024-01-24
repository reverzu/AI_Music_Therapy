from IPython import display as ipd
from audiocraft.models import musicgen
from audiocraft.utils.notebook import display_audio
import torch
import torchaudio
import os
import random
from tqdm import tqdm  # Import tqdm for the progress bar

# Defining the output Directory
output_directory = r"./aiGeneratedMusic/"

model = musicgen.MusicGen.get_pretrained('medium', device='cuda')
model.set_generation_params(duration=10)

prompted_list = ['sad indian acoustic soft, with earthly feel']

res = model.generate(prompted_list, progress=True)

num = int(random.random()*10**10)

# Create a tqdm progress bar
for i, audio in enumerate(tqdm(res, desc='Generating and Saving Audio')):
    audio_cpu = audio.cpu()
    file_path = os.path.join(
        output_directory, f'{prompted_list[i].split(" ")[0]}_audio-{num}.wav')
    torchaudio.save(file_path, audio_cpu, sample_rate=32000)

# Display the saved audio files
for i in range(len(res)):
    file_path = os.path.join(
        output_directory, f'{prompted_list[i].split(" ")[0]}_audio-{num}.wav')
    audio, sample_rate = torchaudio.load(file_path)
    display_audio(audio, sample_rate=sample_rate)
