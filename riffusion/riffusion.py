"""
Command line interface to riffusion.
"""

import base64
import dataclasses
import functools
import argparse
import logging
import io
import json
from pathlib import Path
import time
import typing as T

import PIL
import torch

from huggingface_hub import hf_hub_download

from .audio import wav_bytes_from_spectrogram_image
from .audio import mp3_bytes_from_wav_bytes
from .datatypes import InferenceInput, PromptInput
from .datatypes import InferenceOutput
from .riffusion_pipeline import RiffusionPipeline

import pyaudio

# Log at the INFO level to both stdout and disk
logging.basicConfig(level=logging.INFO)
logging.getLogger().addHandler(logging.FileHandler("server.log"))

# Global variable for the model pipeline
MODEL = None

# Where built-in seed images are stored
SEED_IMAGES_DIR = Path(Path(__file__).resolve().parent.parent, "seed_images")

def setup(checkpoint: str = "riffusion/riffusion-model-v1"):
    # Initialize the model
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

# TODO(hayk): Enable cache here.
# @functools.lru_cache()
def compute(inputs: InferenceInput) -> str:
    """
    Does all the heavy lifting of the request.
    """
    # Load the seed image by ID
    init_image_path = Path(SEED_IMAGES_DIR, f"{inputs.seed_image_id}.png")
    if not init_image_path.is_file():
        return f"Invalid seed image: {inputs.seed_image_id}", 400
    init_image = PIL.Image.open(str(init_image_path)).convert("RGB")

    # Load the mask image by ID
    if inputs.mask_image_id:
        mask_image_path = Path(SEED_IMAGES_DIR, f"{inputs.mask_image_id}.png")
        if not mask_image_path.is_file():
            return f"Invalid mask image: {inputs.mask_image_id}", 400
        mask_image = PIL.Image.open(str(mask_image_path)).convert("RGB")
    else:
        mask_image = None

    # Execute the model to get the spectrogram image
    image = MODEL.riffuse(inputs, init_image=init_image, mask_image=mask_image)

    # Reconstruct audio from the image
    wav_bytes, duration_s = wav_bytes_from_spectrogram_image(image)
    mp3_bytes = mp3_bytes_from_wav_bytes(wav_bytes)

    # Compute the output as base64 encoded strings
    image_bytes = image_bytes_from_image(image, mode="JPEG")

    # Assemble the output dataclass
    output = (wav_bytes, duration_s, mp3_bytes, image_bytes)
    return output


def image_bytes_from_image(image: PIL.Image, mode: str = "PNG") -> io.BytesIO:
    """
    Convert a PIL image into bytes of the given image format.
    """
    image_bytes = io.BytesIO()
    image.save(image_bytes, mode)
    image_bytes.seek(0)
    return image_bytes


def base64_encode(buffer: io.BytesIO) -> str:
    """
    Encode the given buffer as base64.
    """
    return base64.encodebytes(buffer.getvalue()).decode("ascii")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=False, default="riffusion/riffusion-model-v1")
    parser.add_argument('--start_prompt', required=False, default="")
    parser.add_argument('--end_prompt', required=False, default="")
    parser.add_argument('--alpha', required=False, type=float, default=0.0)
    parser.add_argument('--num_inference_steps', type=int, required=False, default=50)
    parser.add_argument('--seed_image_id', required=False, default="og_beat")
    parser.add_argument('--random_seed', type=int, required=False, default=int(time.time()))
    parser.add_argument('--denoising', type=float, required=False, default=0.75)
    parser.add_argument('--guidance', type=float, required=False, default=7.0)
    parser.add_argument('--mask_image_id', required=False, default=None)
    parser.add_argument('--wav_file_out', required=False, default="output.wav")
    parser.add_argument('--play_song', type=bool, default=True, required=False)
    parser.add_argument('--generation_loops', type=int, default=1)
    parser.add_argument('--playback_loops', type=int, default=1)
    args = parser.parse_args()
    setup(args.checkpoint)
    for outer_loop in range(0, args.generation_loops):
        inf_input = InferenceInput(PromptInput(args.start_prompt, args.random_seed + outer_loop, args.denoising, args.guidance), PromptInput(args.end_prompt, args.random_seed + outer_loop, args.denoising, args.guidance), args.alpha, args.num_inference_steps, args.seed_image_id, args.mask_image_id)
        (wav_bytes, duration_s, mp3_bytes, image_bytes) = compute(inf_input)
        wav_bytes.seek(0)
        # Open a file for writing
        with open(args.wav_file_out, 'wb') as f:
            # Write the contents of the BytesIO object to the file
            f.write(wav_bytes.read())

        # Close the BytesIO object
        wav_bytes.close()

        if args.play_song:
            import wave
            p = pyaudio.PyAudio()
            for loop in range(0, args.playback_loops):
                wf = wave.open(args.wav_file_out, 'rb')
                stream = p.open(
                        format = p.get_format_from_width(wf.getsampwidth()),
                        channels = wf.getnchannels(),
                        rate = wf.getframerate(),
                        output = True
                    )
                chunk = 1024
                data = wf.readframes(chunk)
                while data != b'':
                    stream.write(data)
                    data = wf.readframes(chunk)
                stream.close()
            p.terminate()