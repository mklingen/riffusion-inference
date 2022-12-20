"""
Simple TKinter graphical interface to riffusion.
"""
import argparse
import dataclasses
import logging
import time
import threading
import tkinter as tk
import typing as T
import wave
from pathlib import Path
from tkinter import filedialog as fd
from tkinter import ttk

import numpy as np
import PIL
import pyaudio
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageTk

from .audio import (Constants, image_from_spectrogram,
                    mp3_bytes_from_wav_bytes, spectrogram_from_waveform, chunk_spectrograms_from_waveform, concat_wav_bytes_multi_spectrogram_images,
                    wav_bytes_from_spectrogram_image)
from .datatypes import InferenceInput, PromptInput
from .riffusion_pipeline import RiffusionPipeline


# Log at the INFO level to both stdout and disk.
logging.basicConfig(level=logging.INFO)
logging.getLogger().addHandler(logging.FileHandler("server.log"))

# Global variable for the model pipeline.
MODEL = None

# Where built-in seed images are stored.
SEED_IMAGES_DIR = Path(Path(__file__).resolve().parent.parent, "seed_images")

def setup(checkpoint: str = "riffusion/riffusion-model-v1"):
    # Initialize the model.
    global MODEL
    MODEL = load_model(checkpoint=checkpoint)

def load_model(checkpoint: str):
    """
    Load the riffusion model pipeline.
    """
    assert torch.cuda.is_available()

    model = RiffusionPipeline.from_pretrained(
        checkpoint,
        revision="main",
        torch_dtype=torch.float16,
        # Disable the NSFW filter, causes incorrect false positives
        safety_checker=lambda images, **kwargs: (images, False),
    ).to("cuda")

    @dataclasses.dataclass
    class UNet2DConditionOutput:
        sample: torch.FloatTensor

    # Using traced unet from hf hub
    unet_file = hf_hub_download(
        "riffusion/riffusion-model-v1", filename="unet_traced.pt", subfolder="unet_traced"
    )
    unet_traced = torch.jit.load(unet_file)

    class TracedUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.in_channels = model.unet.in_channels
            self.device = model.unet.device
            self.dtype = torch.float16

        def forward(self, latent_model_input, t, encoder_hidden_states):
            sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
            return UNet2DConditionOutput(sample=sample)

    model.unet = TracedUNet()

    model = model.to("cuda")

    return model

def maybe_load_default_image(seed_image_id):
    """Either loads the default image, or raises a FileNotFoundError."""
    init_image_path = Path(SEED_IMAGES_DIR, f"{seed_image_id}.png")
    if not init_image_path.is_file():
        raise FileNotFoundError(f"Invalid seed image: {seed_image_id}")
    return PIL.Image.open(str(init_image_path)).convert("RGB")

# TODO(hayk): Enable cache here.
# @functools.lru_cache()
def compute(inputs: InferenceInput, init_image: PIL.Image) -> PIL.Image:
    """
    Does the actual diffusion, and returns the generated image.
    """
    # Load the seed image by ID
    mask_image = None

    # Execute the model to get the spectrogram image
    return MODEL.riffuse(inputs, init_image=init_image, mask_image=mask_image)

class WaveformInfo:
    def __init__(self, image, compute_mp3=True):
        """Manages a waveform, computing it from an image. Optionally computes MP3 bytes."""
        self.wav_bytes, self.duration_s = wav_bytes_from_spectrogram_image(image)
        if compute_mp3:
            self.mp3_bytes = mp3_bytes_from_wav_bytes(self.wav_bytes)
        else:
            self.mp3_bytes = None

class MultiWaveformInfo:
    def __init__(self, images : list, compute_mp3=True):
        """Manages a waveform, computing it from multiple images. Optionally computes MP3 bytes."""
        self.wav_bytes = concat_wav_bytes_multi_spectrogram_images(images, Constants())
        if compute_mp3:
            self.mp3_bytes = mp3_bytes_from_wav_bytes(self.wav_bytes)
        else:
            self.mp3_bytes = None

class SpectrumImage:
    def __init__(self, img_size=(512, 512), n_images=1, init_color="red"):
        """Manages a spectrograph image, and provides hooks to convert to wav or mp3."""
        self.img_size = img_size
        self.setup_num_images(n_images)
        # TKinter label holding the image.
        self.label = None

    def setup_num_images(self, n_images):
        self.n_images = n_images
        # PIL representation of the image.
        self.pil_imgs = []
        for i in range(0, n_images):
            self.pil_imgs.append(Image.new("RGB", self.img_size, "red"))
        self.concat_img = Image.new("RGB", (self.img_size[1] * self.n_images, self.img_size[0]), "red")
        self.fused_img = self.concat_img.resize(self.img_size)
        # Tkinter representation of the image.
        self.tk_img = ImageTk.PhotoImage(self.fused_img)

    def create_label(self, root : tk.Widget):
        """Creates a tkinter label for the image."""
        self.label = tk.Label(root, image=self.tk_img)
        return self.label

    def copy_from(self, imgs : list):
        """Deep copies the data from another PIL image and computes a waveform."""
        if len(imgs) != len(self.pil_imgs):
            self.setup_num_images(len(imgs))
        for i in range(0, len(imgs)):
            self.pil_imgs[i].paste(imgs[i], (0,0))
            self.concat_img.paste(imgs[i], (i * self.img_size[1], 0))
        self.fused_img = self.concat_img.resize(self.img_size)
        self.tk_img.paste(self.fused_img)
        if self.label:
            self.label["image"] = self.tk_img
            self.label.configure()
        self.waveform = self.compute_waveform()

    def compute_waveform(self):
        """Computes and returns the waveform."""
        return MultiWaveformInfo(self.pil_imgs, compute_mp3=False)

    def on_play_pressed(self, audio_manager):
        """Callback for when the play button is pressed."""
        waveform = self.waveform
        # Ensure the waveform starts at 0.
        waveform.wav_bytes.seek(0)
        # Make a fake file handle so we can stream it.
        wf = wave.open(waveform.wav_bytes, 'rb')
        # Stream to pyaudio.
        stream = audio_manager.open(
                format = audio_manager.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = wf.getframerate(),
                output = True
            )
        # Stream 1024 bytes at a time.
        chunk = 1024
        data = wf.readframes(chunk)
        while data != b'':
            stream.write(data)
            data = wf.readframes(chunk)
        stream.close()

    def save_wav(self, filename : str):
        """Saves the waveform to the given filename"""
        waveform = self.waveform
        waveform.wav_bytes.seek(0)
        with open(filename, 'wb') as f:
            f.write(waveform.wav_bytes.getbuffer())

    def load_wav(self, filename : str):
        """Loads the waveform from the given file name."""
        with wave.open(filename, 'rb') as wav_file:
            # Read the wave file properties
            num_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()
            
            # Read the entire wave file into memory
            constants = Constants()
            if sample_rate != constants.sample_rate:
                raise ValueError("Unfortunately, only sample rates of {} are supported".format(constants.sample_rate))
            
            # Get samples.
            data = np.frombuffer(wav_file.readframes(num_frames), dtype=np.int16)
            # Convert to spectrogram, then to image.
            spectros = chunk_spectrograms_from_waveform(data, constants)
            if len(spectros) != self.n_images:
                self.setup_num_images(len(spectros))
            imgs = []
            for spectro in spectros:
                spectro_img = image_from_spectrogram(spectro)
                imgs.append(spectro_img)
            self.copy_from(imgs)

class LabeledTextEditor:
    def __init__(self, name : str, parent : tk.Widget):
        """TKinter widget with a label and a text editor."""
        self.container = ttk.Frame(parent)
        self.label=ttk.Label(self.container, text=name)
        self.label.pack(side="left")

        self.entry_text=tk.StringVar(None)
        self.entry=ttk.Entry(self.container,textvariable=self.entry_text,width=50)
        self.entry.pack(side="left")

class LabeledSlider:
    def __init__(self, name, default_value : float, min_value : float, max_value : float, parent : tk.Widget):
        """Tkinter widget with a label and a slider."""
        self.container = ttk.Frame(parent)
    
        self.label=ttk.Label(self.container,  text=name)
        self.label.pack(side="left")
        self.variable = tk.DoubleVar(value=default_value)
        # Updates the string label whenever the slider is moved.
        def update_label(event):
            self.value_var.set("{:.2f}".format(self.slider.getdouble(self.slider.get())))
            self.value_label.configure()
        self.slider = ttk.Scale(self.container, orient='horizontal', from_=min_value, to_=max_value, variable=self.variable, command=update_label)
        self.slider.pack(side="left")
        self.value_var = tk.StringVar(value=str(default_value))
        self.value_label = ttk.Label(self.container, textvariable=self.value_var)
        self.value_label.pack(side="left")
        self.slider.bind("<B1-Motion>", update_label)

class LabeledIntSpinBox:
    def __init__(self, name, default_value : int, min_value : int, max_value : int, parent : tk.Widget):
        """TKinter widget with a label and a spin box with an int."""
        self.container = ttk.Frame(parent)
    
        self.label=ttk.Label(self.container,  text=name)
        self.label.pack(side="left")
        self.variable = tk.IntVar(value=default_value)
        def on_spin():
            self.variable.set(self.spinbox.getint(self.spinbox.get()))
        self.spinbox = ttk.Spinbox(self.container, from_=min_value, to=max_value, command=on_spin)
        self.spinbox.set(default_value)
        self.spinbox.pack(side="left")

class RiffusionGUI:
    def __init__(self, checkpoint, default_img_key=None, img_size=(512,512)):
        """Main class representing the GUI."""
        self.img_size = img_size
        self.checkpoint = checkpoint
        self.create_gui()
        self.audio_manager = pyaudio.PyAudio()
        # For long running background operations.
        self.background_thread = None
        # Load the default image.
        if default_img_key:
            self.seed_image.copy_from([maybe_load_default_image(default_img_key)])
            self.generated_image.copy_from(self.seed_image.pil_imgs)

    def set_enabled(self, enabled : bool):
        """Enables/disables the various widgets. This is used during long running background operations."""
        def toggle_button(button):
            if enabled:
                button["state"] = tk.NORMAL
            else:
                button["state"] = tk.DISABLED
        for button in [self.seed_play_button, 
            self.generate_button, 
            self.generated_play_button, 
            self.load_button, 
            self.save_button,
             self.accept_button]:
            toggle_button(button)

    def do_in_background(self, fn):
        """Run a background thread executing the given function. Disables controls while running."""
        def wrapper():
            self.set_enabled(False)
            fn()
            self.set_enabled(True)

        if self.background_thread:
            self.background_thread.join()
        self.background_thread = threading.Thread(target=wrapper)
        self.background_thread.start()
        

    def create_gui(self):
        """Create the GUI elements."""
        self.root = tk.Tk()
        # Create root and top elements.
        self.root.title("Riffusion")
        self.root.geometry("1280x1024")
        self.seed_image = SpectrumImage(self.img_size)
        self.generated_image = SpectrumImage(self.img_size)
        self.seed_label = tk.Label(self.root , text="Seed")
        self.generated_label = tk.Label(self.root , text="Generated")
        self.seed_display = self.seed_image.create_label(self.root)
        self.generated_display = self.generated_image.create_label(self.root)
        self.seed_play_button = tk.Button(self.root , text="Play", command=lambda:self.do_in_background(lambda:self.seed_image.on_play_pressed(self.audio_manager)))
        self.generated_play_button = tk.Button(self.root , text="Play", command=lambda:self.do_in_background(lambda:self.generated_image.on_play_pressed(self.audio_manager)))
        
        # Create a Frame widget for the bottom elements.
        self.controls_frame = ttk.Frame(self.root)

        # Create the form elements for prompts, etc.
        self.prompt_1 = LabeledTextEditor("Prompt A", self.controls_frame)
        self.prompt_2 = LabeledTextEditor("Prompt B", self.controls_frame)
        self.alpha_slider = LabeledSlider("Blend", default_value=0.5, min_value=0, max_value=1, parent=self.controls_frame)
        self.guidance_slider = LabeledSlider("Guidance", default_value=7, min_value=0, max_value=15, parent=self.controls_frame)
        self.denoising_slider = LabeledSlider("Denoising", default_value=0.7, min_value=0, max_value=1, parent=self.controls_frame)
        self.seed_spinner = LabeledIntSpinBox("Seed", default_value=int(time.time()) % 9999, min_value=0, max_value=9999, parent=self.controls_frame)
        self.iters_spinner = LabeledIntSpinBox("Iters", default_value=50, min_value=1, max_value=200, parent=self.controls_frame)
        self.generate_button = ttk.Button(self.controls_frame, text="Generate", command=lambda:self.do_in_background(self.generate))
        self.load_button = ttk.Button(self.controls_frame, text="Load", command=self.load)
        self.accept_button = ttk.Button(self.controls_frame, text="Accept", command=self.accept)
        self.save_button = ttk.Button(self.controls_frame, text="Save", command=self.save)


        # Pack the widgets in the Frame.
        self.prompt_1.container.pack()
        self.prompt_2.container.pack()
        self.alpha_slider.container.pack()
        self.guidance_slider.container.pack()
        self.denoising_slider.container.pack()
        self.seed_spinner.container.pack()
        self.iters_spinner.container.pack()
        self.generate_button.pack()
        self.load_button.pack()
        self.accept_button.pack()
        self.save_button.pack()

        # Use the grid geometry manager to position the widgets.
        self.seed_label.grid(row=0,column=0)
        self.seed_display.grid(row=1,column=0)
        self.seed_play_button.grid(row=2, column=0)
        self.generated_label.grid(row=0, column=1)
        self.generated_display.grid(row=1, column=1)
        self.generated_play_button.grid(row=2, column=1)
        self.controls_frame.grid(row=3, column=0, columnspan=2, sticky='nsew')

    def load(self):
        """Loads a .wav file from disk."""
        filename = fd.askopenfilename(defaultextension=".wav", filetypes=(('Wave File', '*.wav'),))
        if filename:
            self.seed_image.load_wav(filename)

    def save(self):
        """Saves a .wav file to disk."""
        filename = fd.asksaveasfilename(defaultextension=".wav", filetypes=(('Wave File', '*.wav'),))
        if filename:
            self.generated_image.save_wav(filename)

    def accept(self):
        """Copies the generated image into the seed image, allowing iteration."""
        self.seed_image.copy_from(self.generated_image.pil_img)

    def generate(self):
        """Generates a new image from the parameters."""
        global MODEL
        if not MODEL:
            setup(self.checkpoint)
        seed = self.seed_spinner.variable.get()
        denoising = self.denoising_slider.variable.get()
        guidance = self.guidance_slider.variable.get()
        alpha = self.alpha_slider.variable.get()
        iters = self.iters_spinner.variable.get()
        gen_imgs = []
        # Compute a new image for every single chunk.
        for img in self.seed_image.pil_imgs:
            inf_input =  InferenceInput(PromptInput(self.prompt_1.entry_text.get(), seed, denoising, guidance), PromptInput(self.prompt_2.entry_text.get(), seed, denoising, guidance), alpha, iters, None, None)
            gen_imgs.append(compute(inf_input, img))

        self.generated_image.copy_from(gen_imgs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=False, default="riffusion/riffusion-model-v1")
    parser.add_argument('--seed_image_id', required=False, default="og_beat")
    args = parser.parse_args()
    gui = RiffusionGUI(args.checkpoint, default_img_key=args.seed_image_id)
    gui.root.mainloop()