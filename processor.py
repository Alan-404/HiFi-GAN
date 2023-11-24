from typing import Any, Optional
import librosa
import numpy as np
import torch

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

        self.mel_filterbank = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=n_mel_channels, fmin=fmin, fmax=fmax, htk=htk)

    def load_audio(self, path: str):
        signal, _ = librosa.load(path, sr=self.sample_rate, mono=False)
        signal = self.standard_scale(signal)
        return signal
    
    def standard_scale(self, x: np.ndarray):
        return (x - x.mean()) / np.sqrt(x.var() + 1e-7)

    def spectral_normalize(self, x, C=1, clip_val=1e-5):
        return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

    def mel_spectrogram(self, signal: np.ndarray):
        stft = librosa.stft(y=signal, n_fft=self.fft_size, hop_length=self.hop_length, win_length=self.window_size, window='hann', center=True, pad_mode='reflect')

        mel_spec = np.sqrt(np.power(np.abs(stft), 2) + (1e-9))

        mel_spec = np.dot(self.mel_filterbank, mel_spec)

        if signal.ndim > 1:
            mel_spec = np.transpose(mel_spec, axes=(1, 0, 2))

        mel_spec = self.spectral_normalize(mel_spec)

        return mel_spec
    
    def __call__(self, signals: list, max_len: Optional[int] = None) -> Any:
        if max_len is None:
            max_len = np.max([len(signal) for signal in signals])

        padded_signals = []
        mels = []

        for signal in signals:
            padded_signal = np.pad(signal, (0, max_len - len(signal)), mode='constant', constant_values=0.0)

            padded_signals.append(padded_signal)
            mels.append(self.mel_spectrogram(padded_signal))

        return np.array(mels), np.array(padded_signals)
    
