import os
import io
import pandas as pd
from glob import glob
from tqdm import tqdm

if os.path.exists("./datasets") == False:
    os.mkdir("./datasets")

os.system('curl -L -o ./datasets/infore_technology.zip "https://huggingface.co/datasets/ntt123/infore/resolve/main/infore_16k_denoised.zip"')
os.system('unzip ./datasets/infore_technology.zip -d ./datasets/infore_technology')

audio_paths = glob("./datasets/infore_technology/data/*.wav")

labels = []

print("Creating ManiFest Dataset...")
for path in tqdm(audio_paths):
    text_path = path.replace(".wav", ".txt")
    text = io.open(text_path).read().strip().split("\n")
    assert len(text) == 1

    labels.append(text[0])

pd.DataFrame({
    'path': audio_paths,
    'text': labels
}).to_csv("./datasets/data.csv", sep="\t", index=None)

print(f"Dataset is saved at ./datasets")