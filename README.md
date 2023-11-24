# (TTS) HiFi GAN Model: Mel - Spectrogram to Waveform
## Model Architecture
<img src="./assets/model.png"/>

## Folder Structure
    assets                      # Assets Folder of README
    config.json                 # Configuration Training
    .gitignore
    processor.py                # Data Handler
    README.md
    requirements.txt
    hifi_gan.py                 # HiFi - GAN Model
    train.py                    # Training File
    trainer.py                  # Trainer for Model

## Setup Dataset
| Name    | Link |
| --------- | ------- |
| InfoRE Tech 16h     | https://huggingface.co/datasets/ntt123/infore/resolve/main/infore_16k_denoised.zip        |

## Setup Environment
```
git clone https://git.cads.live/trind18/hifi-gan.git
cd hifi-gan
python3 -m venv tts
source tts/bin/activate
pip install -r requirements.txt
```

## Train Model
```
python3 train.py --config_path ./config/config.json
```