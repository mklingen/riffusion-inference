"""
Inference server for the riffusion project.
"""

import base64
import dataclasses
import functools
from scipy.io import wavfile
import logging
import io
import json
from pathlib import Path
import time
import typing as T
import wave

import dacite
import flask

from flask_cors import CORS
import PIL
import numpy as np
import torch

from huggingface_hub import hf_hub_download

from .audio import Constants, base64_encode, chunk_spectrograms_from_waveform, concat_wav_bytes_multi_spectrogram_images, image_from_spectrogram, wav_bytes_from_spectrogram_image, wav_from_base_64
from .audio import mp3_bytes_from_wav_bytes
from .datatypes import PromptInput, VstInput, InferenceInput
from .datatypes import VstOutput
from .riffusion_pipeline import RiffusionPipeline

# Flask app with CORS
app = flask.Flask(__name__)
CORS(app)

# Log at the INFO level to both stdout and disk
logging.basicConfig(level=logging.INFO)
logging.getLogger().addHandler(logging.FileHandler("server.log"))

# Global variable for the model pipeline
MODEL = None

# Where built-in seed images are stored
SEED_IMAGES_DIR = Path(Path(__file__).resolve().parent.parent, "seed_images")


def run_app(
    *,
    checkpoint: str = "riffusion/riffusion-model-v1",
    host: str = "127.0.0.1",
    port: int = 3000,
    debug: bool = False,
    ssl_certificate: T.Optional[str] = None,
    ssl_key: T.Optional[str] = None,
):
    """
    Run a flask API that serves the given riffusion model checkpoint.
    """
    # Initialize the model
    global MODEL
    MODEL = load_model(checkpoint=checkpoint)

    args = dict(
        debug=debug,
        threaded=False,
        host=host,
        port=port,
    )

    if ssl_certificate:
        assert ssl_key is not None
        args["ssl_context"] = (ssl_certificate, ssl_key)

    app.run(**args)


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


@app.route("/run_vst/", methods=["POST"])
def run_vst():
    """
    Execute the riffusion model as an API.

    Inputs:
        Serialized JSON of the VstInput dataclass

    Returns:
        Serialized JSON of the VstOutput dataclass
    """
    start_time = time.time()
    # Parse the payload as JSON
    json_data = json.loads(flask.request.data)

    # Log the request
    #logging.info(json_data)

    # Parse an VstInput dataclass from the payload
    try:
        inputs = dacite.from_dict(VstInput, json_data)
    except dacite.exceptions.WrongTypeError as exception:
        logging.info(json_data)
        return str(exception), 400
    except dacite.exceptions.MissingValueError as exception:
        logging.info(json_data)
        return str(exception), 400

    response = compute(inputs)

    # Log the total time
    logging.info(f"Request took {time.time() - start_time:.2f} s")

    return response


# TODO(hayk): Enable cache here.
# @functools.lru_cache()
def compute(inputs: VstInput) -> str:
    """
    Does all the heavy lifting of the request.
    """
    # Load the mask image by ID
    if inputs.mask_image_id:
        mask_image_path = Path(SEED_IMAGES_DIR, f"{inputs.mask_image_id}.png")
        if not mask_image_path.is_file():
            return f"Invalid mask image: {inputs.mask_image_id}", 400
        mask_image = PIL.Image.open(str(mask_image_path)).convert("RGB")
    else:
        mask_image = None
    raw_wav_bytes_base64 = inputs.audio
    raw_wav_bytes = wav_from_base_64(raw_wav_bytes_base64)
    wav_bytes_input = io.BytesIO(raw_wav_bytes)
    wav_bytes_input.seek(0)
    constants = Constants()
    wav_bytes_input.seek(0)
    with wave.open(wav_bytes_input, 'rb') as wav_file:
        # Read the wave file properties
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        # Read the entire wave file into memory
        if sample_rate != constants.sample_rate:
            raise ValueError("Unfortunately, only sample rates of {} are supported".format(constants.sample_rate))
        
        if num_frames == 0:
            num_frames = len(raw_wav_bytes) / sample_width

        # Get samples.
        input_data = np.frombuffer(wav_file.readframes(num_frames), dtype=np.int16)
    
    logging.info("Waveform had {} samples. Input rate was {}".format(input_data.shape[0], sample_rate))
    spectro_images = chunk_spectrograms_from_waveform(input_data.astype(np.float16), constants)
    gen_images = []
    gen_input = InferenceInput(inputs.start, inputs.end, inputs.alpha, inputs.num_inference_steps, None, None)
    for init_spectro in spectro_images:
        init_image = image_from_spectrogram(init_spectro)
        # Execute the model to get the spectrogram image
        gen_images.append(MODEL.riffuse(gen_input, init_image=init_image, mask_image=mask_image))

    # Reconstruct audio from the image
    wav_bytes = concat_wav_bytes_multi_spectrogram_images(gen_images, constants)
    wav_bytes.seek(0)
    wav_bytes_input.seek(0)
    # Assemble the output dataclass
    output = VstOutput(
        audio=base64_encode(wav_bytes)
    )

    resp =  flask.jsonify(dataclasses.asdict(output))
    return resp

if __name__ == "__main__":
    import argh
    argh.dispatch_command(run_app)
