"""
Audio processing tools to convert between spectrogram images and waveforms.
"""
import io
import typing as T

import numpy as np
from PIL import Image
import pydub
from scipy.io import wavfile
import torch
import torchaudio
import matplotlib.pyplot as plt
import math


class Constants:
    def __init__(self):
        self.max_volume = 50
        self.power_for_image = 0.25
        self.sample_rate = 44100  # [Hz]
        self.clip_duration_ms = 5000  # [ms]

        self.bins_per_image = 512
        self.n_mels = 512

        # FFT parameters
        self.window_duration_ms = 100  # [ms]
        self.padded_duration_ms = 400  # [ms]
        self.step_size_ms = 10  # [ms]

        self.n_fft = int(self.padded_duration_ms / 1000.0 * self.sample_rate)
        self.hop_length = int((self.sample_rate  * self.get_clip_length_seconds()) / self.bins_per_image) #int(self.step_size_ms / 1000.0 * self.sample_rate)
        self.win_length = int(self.window_duration_ms / 1000.0 * self.sample_rate)
        self.num_samples = int(math.ceil(self.get_clip_length_seconds() * self.sample_rate))
        assert int(self.num_samples / self.hop_length) == self.bins_per_image
    
    def get_clip_length_seconds(self):
        return self.clip_duration_ms / 1000.0

def concat_wav_bytes_multi_spectrogram_images(images : list, constants : Constants) -> io.BytesIO:
    """
    Reconstruct a WAV audio clip from multiple spectrogram images..
    """
    samples_concat = None
    print("Concatenating {} images".format(len(images)))
    for image in images:
        Sxx = spectrogram_from_image(image, max_volume=constants.max_volume, power_for_image=constants.power_for_image)

        samples = waveform_from_spectrogram(
            Sxx=Sxx,
            n_fft=constants.n_fft,
            hop_length=constants.hop_length,
            win_length=constants.win_length,
            num_samples=constants.num_samples,
            sample_rate=constants.sample_rate,
            mel_scale=True,
            n_mels=constants.n_mels,
            max_mel_iters=200,
            num_griffin_lim_iters=32,
        )
        if samples_concat is None:
            samples_concat = samples
        else:
            samples_concat = np.concatenate([samples_concat, samples])

    wav_bytes = io.BytesIO()
    wavfile.write(wav_bytes, constants.sample_rate, samples_concat.astype(np.int16))
    wav_bytes.seek(0)
    return wav_bytes

def wav_bytes_from_spectrogram_image(image: Image.Image, constants=None, show_waveform=False, show_spectrogram=False) -> T.Tuple[io.BytesIO, float]:
    """
    Reconstruct a WAV audio clip from a spectrogram image. Also returns the duration in seconds.
    """
    if show_spectrogram:
        image.show()
    if not constants:
        constants = Constants()
    Sxx = spectrogram_from_image(image, max_volume=constants.max_volume, power_for_image=constants.power_for_image)

    samples = waveform_from_spectrogram(
        Sxx=Sxx,
        n_fft=constants.n_fft,
        hop_length=constants.hop_length,
        win_length=constants.win_length,
        num_samples=constants.num_samples,
        sample_rate=constants.sample_rate,
        mel_scale=True,
        n_mels=constants.n_mels,
        max_mel_iters=200,
        num_griffin_lim_iters=32,
    )
    if show_waveform:
        plt.plot(samples)
        plt.show()
    wav_bytes = io.BytesIO()
    wavfile.write(wav_bytes, constants.sample_rate, samples.astype(np.int16))
    wav_bytes.seek(0)

    duration_s = float(len(samples)) / constants.sample_rate

    return wav_bytes, duration_s

def image_from_spectrogram(
    data: np.ndarray, max_volume: float = 50, power_for_image: float = 0.25
) -> Image.Image:
    """
    Compute a spectrogram magnitude array from a spectrogram image.

    TODO(hayk): Add image_from_spectrogram and call this out as the reverse.
    """
    # Reverse the power curve
    data = np.power(data.astype(np.float64), power_for_image).astype(np.float32)
    # Rescale to max volume
    data = data * (1.0 / (max_volume / 255))

    # Invert
    data = 255 - data

    # Flip Y take a single channel
    data = data[::-1, :]
    return Image.fromarray(data).convert("RGB")

def spectrogram_from_image(
    image: Image.Image, max_volume: float = 50, power_for_image: float = 0.25
) -> np.ndarray:
    """
    Compute a spectrogram magnitude array from a spectrogram image.

    TODO(hayk): Add image_from_spectrogram and call this out as the reverse.
    """
    # Convert to a numpy array of floats
    data = np.array(image).astype(np.float64)

    # Flip Y take a single channel
    data = data[::-1, :, 0]

    # Invert
    data = 255 - data

    # Rescale to max volume
    data = data * max_volume / 255

    # Reverse the power curve
    data = np.power(data, 1 / power_for_image)
    return data

def image_from_spectrogram(
    spectrogram: np.ndarray, max_volume: float = 50, power_for_image: float = 0.25
) -> Image.Image:
    """
    Compute a spectrogram image from a spectrogram magnitude array.
    """
    # Apply the power curve
    data = np.power(spectrogram, power_for_image)
    
    # Rescale to 0-1
    data = data / np.max(data)

    # Rescale to 0-255
    data = data * 255

    # Invert
    data = 255 - data

    # Convert to a PIL image
    image = Image.fromarray(data.astype(np.uint8))

    # Flip Y
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Convert to RGB
    image = image.convert("RGB")

    return image

def chunk_waveform(waveform: np.ndarray, constants : Constants):
    """Takes in a waveform (for example, from a .wav file) expressed as a numpy array 
    of samples that are 16 bit, and a sample rate (in Hz), and outputs a series of chunks 
    that are precisely 5 seconds long covering the waveform. If the waveform is less than 5 seconds long,
    it will be looped from the beginning. If the final chunk does not fit cleanly into 5 seconds, the start
    of the waveform will be looped in the final chunk."""
    # Calculate the number of samples in 5 seconds
    num_samples = constants.num_samples
    total_samples = waveform.shape[0]
    # Calculate the number of complete 5-second chunks in the waveform.
    num_chunks = int(math.ceil(total_samples / num_samples))
    if total_samples < num_samples:
        num_chunks = 1
        print("Fewer samples than expected. File is short. Will try to pad it.")
    # Initialize empty list for storing the chunks.
    chunks = []

    # Loop through the waveform, extracting 5-second chunks.
    for i in range(num_chunks):
        print("Chunking from {} to {}".format(min(i*num_samples, total_samples - 1), min((i+1)*num_samples, total_samples)))
        chunk = waveform[min(i*num_samples, total_samples - 1):min((i+1)*num_samples, total_samples)]
        print("Resulting chunk had length {}".format(chunk.shape[0]))
        if chunk.shape[0] < num_samples:
            diff = min(num_samples - chunk.shape[0], total_samples)
            if (num_samples - diff) / num_samples < 0.25:
                print("Final chunk was too small. Deleting the rest of the clip.")
                break
            print("Appending {} padding samples. Chunk size is {}/{}".format(diff, chunk.shape[0], num_samples))
            chunk = np.concatenate([chunk, np.zeros(diff,)])
        assert chunk.shape[0] == num_samples
        chunks.append(chunk)

    print("Should be {} chunk(s) in this file".format(len(chunks)))
    return chunks

def chunk_spectrograms_from_waveform(waveform: np.ndarray, constants : Constants):
    """Takes in a waveform (for example, from a .wav file) expressed as a numpy array 
    of samples that are 16 bit, and a sample rate (in Hz), and outputs a series of chunks 
    that are precisely 5 seconds long covering the waveform as a spectrogram. If the waveform is less than 5 seconds long,
    it will be looped from the beginning. If the final chunk does not fit cleanly into 5 seconds, the start
    of the waveform will be looped in the final chunk."""
    # Initialize to empty.
    spectrograms = []
    # Convert each chunk into a spectrogram.
    for chunk in chunk_waveform(waveform, constants):
        spectrograms.append(spectrogram_from_waveform(chunk, constants.sample_rate, constants.n_fft, constants.hop_length, constants.win_length, True, constants.n_mels))
    return spectrograms

def spectrogram_from_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    mel_scale: bool = True,
    n_mels: int = 512,
) -> np.ndarray:
    """
    Compute a spectrogram from a waveform.
    """
    spectrogram_func = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        power=None,
        hop_length=hop_length,
        win_length=win_length,
    )

    waveform_tensor = torch.from_numpy(waveform.astype(np.float32)).reshape(1, -1)
    Sxx_complex = spectrogram_func(waveform_tensor).numpy()[0]

    Sxx_mag = np.abs(Sxx_complex)

    if mel_scale:
        mel_scaler = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=0,
            f_max=10000,
            n_stft=n_fft // 2 + 1,
            norm=None,
            mel_scale="htk",
        )
        Sxx_mag = mel_scaler(torch.from_numpy(Sxx_mag)).numpy()
    np.resize(Sxx_mag, (n_mels, n_mels))
    return Sxx_mag


def waveform_from_spectrogram(
    Sxx: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    num_samples: int,
    sample_rate: int,
    mel_scale: bool = True,
    n_mels: int = 512,
    max_mel_iters: int = 200,
    num_griffin_lim_iters: int = 32,
    device: str = "cuda:0",
) -> np.ndarray:
    """
    Reconstruct a waveform from a spectrogram.

    This is an approximate inverse of spectrogram_from_waveform, using the Griffin-Lim algorithm
    to approximate the phase.
    """
    Sxx_torch = torch.from_numpy(Sxx).to(device).to(torch.float32)
    
    # TODO(hayk): Make this a class that caches the two things

    if mel_scale:
        mel_inv_scaler = torchaudio.transforms.InverseMelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=0,
            f_max=10000,
            n_stft=n_fft // 2 + 1,
            norm=None,
            mel_scale="htk",
            max_iter=max_mel_iters,
        ).to(device)

        Sxx_torch = mel_inv_scaler(Sxx_torch)

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=1.0,
        n_iter=num_griffin_lim_iters,
    ).to(device)

    waveform = griffin_lim(Sxx_torch).cpu().numpy()

    return waveform


def mp3_bytes_from_wav_bytes(wav_bytes: io.BytesIO) -> io.BytesIO:
    mp3_bytes = io.BytesIO()
    sound = pydub.AudioSegment.from_wav(wav_bytes)
    sound.export(mp3_bytes, format="mp3")
    mp3_bytes.seek(0)
    return mp3_bytes
