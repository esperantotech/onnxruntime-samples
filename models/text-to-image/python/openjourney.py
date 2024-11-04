#!/usr/bin/env python3

import sys
import os
import numpy as np
import onnx
import onnxruntime as ort
from argparse import ArgumentParser, Namespace
from typing import Sequence, Optional
from pathlib import Path
from transformers import CLIPTokenizer
from diffusers import PNDMScheduler
from tqdm.auto import tqdm
import subprocess
import torch
import time
from PIL import Image


# Import utils.py
sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from common import txt2img

def run_txt_clip(session, input_ids):

    in_name_input_ids = session.get_inputs()[0].name

    start = time.time()
    text_embeddings = session.run([], {in_name_input_ids: input_ids})[0] 
    end = time.time()
    diff_time = np.round((end-start)*1000,2)
    print(f'Time to run txt_clip tooks {str(diff_time)}ms)')

    return text_embeddings


def txt_clip(model : Path, args : ArgumentParser, tokenizer : any) -> any:

    sess_options = ort.SessionOptions()
    batch_size = len([args.prompt])

    poptions_txt_clip = {
        "etglow_greedy": "true",
        "etglow_onnx_shape_params": f'batch={batch_size};sequence={tokenizer.model_max_length}'
    }

    if args.provider == "etglow":
        session = ort.InferenceSession(model, sess_options, providers = ['EtGlowExecutionProvider'], provider_options = [poptions_txt_clip])
    else:
        session = ort.InferenceSession(model)    

    text_input = tokenizer([args.prompt], padding="max_length", max_length=tokenizer.model_max_length, return_tensors = "pt")
    input_ids = text_input.input_ids.numpy()
    input_ids = input_ids.astype(np.int32)

    text_embeddings = run_txt_clip(session, input_ids)

    max_length = text_input.input_ids.shape[-1]
    text_uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_input = text_uncond_input.input_ids.numpy()
    uncond_input = uncond_input.astype(np.int32)

    uncond_embeddings = run_txt_clip(session, uncond_input)     

    text_embeddings = torch.cat([torch.from_numpy(uncond_embeddings), torch.from_numpy(text_embeddings)])

    return text_embeddings

def run_unet(session, t, latent_model_input, text_embeddings):

    in_name_sample = session.get_inputs()[0].name
    in_name_timestep = session.get_inputs()[1].name
    in_name_encoder_hidden_states = session.get_inputs()[2].name

    start = time.time()
    noise_pred = session.run([], {in_name_sample: latent_model_input,
                                  in_name_timestep: [np.array(t)],
                                  in_name_encoder_hidden_states: text_embeddings})[0]
    end = time.time()
    diff_time = np.round((end - start) * 1000, 2)
    print(f'Time to run unet tooks {str(diff_time)}ms) at iteration {t}')

    return noise_pred


def unet(model : Path, args : ArgumentParser, tokenizer : any, height : int, width: int) -> any:

    sess_options = ort.SessionOptions()
    batch_size = len([args.prompt])    

    poptions_unet = {
        "etglow_greedy": "true",
        "etglow_onnx_shape_params": f'batch={batch_size*2};channels=4;height={height//8};width={width//8};sequence={tokenizer.model_max_length}'
    }    

    if args.provider == "etglow":
        session = ort.InferenceSession(model, sess_options, providers = ['EtGlowExecutionProvider'], provider_options = [poptions_unet])
    else:
        session = ort.InferenceSession(model)

    return session

def run_vae_decode(session, latents):
    
    in_name_latent_sample = session.get_inputs()[0].name

    start = time.time()   
    image = session.run([], {in_name_latent_sample: latents})
    end = time.time()
    diff_time = np.round((end-start)*1000,2)
    print(f'Time to run vae_decode tooks {str(diff_time)}ms)')

    return image

def vae_decode(model : Path, args : ArgumentParser, height : int, width : int) -> any:

    sess_options = ort.SessionOptions()
    batch_size = len([args.prompt])    
    poptions_vae_decode = {
        "etglow_greedy": "true",
        "etglow_onnx_shape_params": f'batch={batch_size};channels=4;height={height//8};width={width//8};channels_image=3;height_image={height};width_image={width}'
    }

    if args.provider == "etglow":
        session =  ort.InferenceSession(model, sess_options, providers = ['EtGlowExecutionProvider'], provider_options = [poptions_vae_decode])
    else:
        session = ort.InferenceSession(model)

    return session

def main(argv: Optional[Sequence[str]] = None):
    """Launch Openjouney text-to-image model."""
    parser = txt2img.get_arg_parser()
    parser.add_argument("-o", "--output", default="image.png", help="Output file image.")
    args = parser.parse_args(argv)

    artifacts_path = Path(args.artifacts)

    model_name = "openjourney-fp32-onnx" if args.precision == "fp32" else 'openjourney-fp16-onnx'

    submodel_name = ['text_encoder', 'unet', 'vae_decoder']
    model_txt_clip_path = artifacts_path / f'models/{model_name}/{submodel_name[0]}/model.onnx'
    model_unet_path = artifacts_path / f'models/{model_name}/{submodel_name[1]}/model.onnx'
    model_vae_decode_path = artifacts_path / f'models/{model_name}/{submodel_name[2]}/model.onnx'

    #it is not the openjourney predefined scheduler
    scheduler = PNDMScheduler(beta_start = 0.00085, beta_end = 0.012, beta_schedule = "scaled_linear",
                              num_train_timesteps = 1000)

    prompt = [args.prompt]
    batch_size = len(prompt)
    height = 512    #default height of stable Diffusion
    width = 512     #default width of stable Diffusion
    num_inference_steps = args.totalInferences     #number of denoising steps
    guidance_scale = 7.5                           #Scale for classifier-free guidance    
    generator = torch.manual_seed(args.randomseed) #Seed generator to create the initial latent noise
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    text_embeddings = txt_clip(model_txt_clip_path, args, tokenizer)

    latents = torch.randn((batch_size, 4, height // 8, width // 8), generator = generator)
    #initialize the scheduler with the inference steps
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma

    if args.precision == "fp16":
        text_embeddings = text_embeddings.numpy().astype(np.float16)
    else:
        text_embeddings = text_embeddings.numpy()

    session_unet = unet(model_unet_path, args, tokenizer, height, width)

    for t in tqdm(scheduler.timesteps):
        #expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep = t)

        if args.precision == "fp16":
            latent_model_input = latent_model_input.numpy().astype(np.float16)
        else:
            latent_model_input = latent_model_input.numpy()
            
        #predict the noise residual
        with torch.no_grad():
            noise_pred = run_unet(session_unet, t, latent_model_input, text_embeddings)

        #perform guidance
        noise_pred = torch.from_numpy(noise_pred) #we convert it to tensor, if not chunk does not work\n",
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        #compute the previous noisiy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    #scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents

    if args.precision == "fp16":
        latents = latents.numpy().astype(np.float16)
    else:
        latents = latents.numpy()

    session_vae_decode = vae_decode(model_vae_decode_path, args, height, width)

    with torch.no_grad():
        image = run_vae_decode(session_vae_decode, latents)

    #Here is assuming batch_size 1 
    if args.precision == "fp16":
        image = image[0].astype(np.float32)
    else:
        image = image[0]

    image = (torch.from_numpy(image) / 2 + 0.5).clamp(0, 1)
    image = image.permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    pil_images[0].save(args.output)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))