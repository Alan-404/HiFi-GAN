from typing import Any, Optional
import librosa
import numpy as np
import torch
from pydub import AudioSegment
import torchaudio
import torch.nn.functional as F

MAX_AUDIO_VALUE = 32768

class HiFiGANProcessor:
    def __init__(self, sample_rate: int = 22050, n_mel_channels: int = 80, fft_size: int = 1024, window_size: int = 1024, hop_length: int = 256, fmax: float = 8000.0, fmin: float = 0.0, htk: bool = True) -> None:

        # Audio
        self.sample_rate = sample_rate
        self.n_mel_channels = n_mel_channels
        self.fft_size = fft_size
        self.window_size = window_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.htk = htk

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=fft_size,
            hop_length=hop_length,
            win_length=window_size,
            f_min=fmin,
            f_max=fmax,
            n_mels=n_mel_channels
        )

    def load_audio(self, path: str):
        audio = AudioSegment.from_file(path).set_frame_rate(self.sample_rate).get_array_of_samples()
        signal = torch.tensor(audio) / MAX_AUDIO_VALUE
        signal = self.standard_normalize(signal)
        return signal
    
    def standard_normalize(self, signal: torch.Tensor):
        return (signal - signal.mean()) / signal.std()
    
    def spec_normalize(self, mel: torch.Tensor, clip_val: float = 1e-5, C: int = 1):
        return torch.log(torch.clamp(mel, min=clip_val) * C)

    def log_mel_spectrogram(self, signal: torch.Tensor):
        mel_spec = self.mel_transform(signal)

        log_mel = self.spec_normalize(mel_spec)

        return log_mel
    
    def __call__(self, signals: list, max_len: Optional[int] = None) -> Any:
        if max_len is None:
            max_len = np.max([len(signal) for signal in signals])

        mels = []
        padded_signals = []
        for signal in signals:
            padded_signal = F.pad(signal, pad=(0, max_len - len(signal)), mode='constant', value=0.0)

            padded_signals.append(padded_signal)

            mels.append(self.log_mel_spectrogram(padded_signal))

        return torch.stack(mels), torch.stack(padded_signals)