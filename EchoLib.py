import numpy as np
import librosa
import matplotlib.pyplot as plt
from typing import Tuple, Callable

class AudioProcessor:
    def __init__(self, filename: str):
        self.filename = filename
        self.audio, self.sr = self._load_audio()
    
    def _load_audio(self) -> Tuple[np.ndarray, int]:
        return librosa.load(f'UrbanSound8K/UrbanSound8K/audio/fold1/{self.filename}.wav', sr=None)
    
    def plot_waveform(self):
        plt.figure(figsize=(15, 3))
        plt.plot(self.audio, lw=0.1, alpha=1)
        plt.show()

class WindowFunction:
    @staticmethod
    def rect(wi: np.ndarray) -> np.ndarray:
        y = np.ones_like(wi)
        y[(wi < 0) | (wi > 1)] = 0
        return y

    @staticmethod
    def hann(wi: np.ndarray) -> np.ndarray:
        y = np.sin(wi * np.pi) ** 2
        y[(wi < 0) | (wi > 1)] = 0
        return y

    @staticmethod
    def hamm(wi: np.ndarray, a0: float = 0.54) -> np.ndarray:
        y = a0 - ((1 - a0) * np.cos(2 * np.pi * wi))
        y[(wi < 0) | (wi > 1)] = 0
        return y

    @staticmethod
    def show_window(window_fn: Callable, n: int = 200):
        plt.figure(figsize=(8, 5))
        xs = np.linspace(0, 1, n, True)
        plt.ylim(0, 1)
        plt.plot(xs, window_fn(xs), lw=1)
        plt.show()

class STFTProcessor:
    def __init__(self, audio: np.ndarray, sr: int, window_fn: Callable):
        self.audio = audio
        self.sr = sr
        self.window_fn = window_fn
    
    def compute_stft(self, wsize_ms: int = 25, hsize_ms: int = None, fixed: bool = True) -> np.ndarray:
        spect = []
        wsize = int(wsize_ms * 1e-3 * self.sr)
        if hsize_ms is None:
            hsize_ms = wsize_ms / 2
        hsize = int(hsize_ms * 1e-3 * self.sr)
        
        window = self.window_fn(np.linspace(0, 1, wsize, True))
        
        i = 0
        while i < len(self.audio):
            print(f'\r{i}/{len(self.audio)}...', end='')
            
            if fixed:
                sliced = self.audio[i : i + wsize] + 0
                if len(sliced) < wsize:
                    sliced = np.concatenate((sliced, np.zeros((wsize - len(sliced),))))
                sliced *= window
            else:
                sliced = np.zeros_like(self.audio)
                sliced[i : i + wsize] = window[:len(sliced[i : i + wsize])]
                sliced *= self.audio
            
            spect.append(np.fft.rfft(sliced))
            i += hsize
        
        spect = np.array(spect, dtype=np.complex64)
        spect = np.transpose(spect)
        spect = spect[::-1, :-1]
        
        return spect

class AudioAnalyzer:
    def __init__(self, filename: str):
        self.processor = AudioProcessor(filename)
    
    def analyze(self):
        rect_spectrogram = STFTProcessor(self.processor.audio, self.processor.sr, WindowFunction.rect).compute_stft()
        hann_spectrogram = STFTProcessor(self.processor.audio, self.processor.sr, WindowFunction.hann).compute_stft()
        hamm_spectrogram = STFTProcessor(self.processor.audio, self.processor.sr, WindowFunction.hamm).compute_stft()
        
        rect_spectrogram = np.log(np.abs(rect_spectrogram)) * 20
        hann_spectrogram = np.log(np.abs(hann_spectrogram)) * 20
        hamm_spectrogram = np.log(np.abs(hamm_spectrogram)) * 20
        
        self._plot_spectrograms(rect_spectrogram, hann_spectrogram, hamm_spectrogram)
    
    def _plot_spectrograms(self, rect_spectrogram, hann_spectrogram, hamm_spectrogram):
        fig, ax = plt.subplots(4, 1, figsize=(15, 8))
        timeax = np.array(range(len(self.processor.audio))) / self.processor.sr
        
        ax[0].plot(timeax, self.processor.audio, alpha=1, lw=0.2)
        ax[0].axis('off')
        ax[0].set_xlim((timeax[0], timeax[-1]))
        ax[0].title.set_text('Audio Waveform')
        
        spectrograms = [rect_spectrogram, hann_spectrogram, hamm_spectrogram]
        titles = ['Rectangular Windowing', 'Hanning Windowing', 'Hamming Windowing']
        
        for i in range(3):
            ax[i + 1].imshow(spectrograms[i], cmap='inferno', aspect=(1e-1 * spectrograms[i].shape[1] / spectrograms[i].shape[0]))
            ax[i + 1].title.set_text(titles[i])
            ax[i + 1].axis('off')
        
        fig.tight_layout()
        plt.show()
