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
import time

# Import utils.py
sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from common import txt2img

def main(argv: Optional[Sequence[str]] = None):
    """Launch Openjouney text-to-image model."""
    parser = txt2img.get_arg_parser()
    args = parser.parse_args(argv)

    artifacts_path = Path(args.artifacts)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    model_name = None
    #It is not the optimized onnx
    if (args.datatype == "fp32"):
        model_name = 'openjourney_onnx_fp32_aws'
    else:
        model_name = 'openjourney_onnx_fp16_aws'
        
    submodel_name = ['text_encoder', 'unet', 'scheduler_scale', 'scheduler_step', 'vae_decoder']

    model_txt_clip_path = artifacts_path / f'models/{model_name}/{submodel_name[0]}/model.onnx'
    model_unet_path = artifacts_path / f'models/{model_name}/{submodel_name[1]}/model.onnx'
    #model_scheduler_scale_path = artifacts_path / f'models/{model_name}/{submodel_name[2]}/model.onnx'
    #model_scheduler_step_path = artifacts_path / f'models/{model_name}/{submodel_name[3]}/model.onnx'
    model_vae_decode_path = artifacts_path / f'models/{model_name}/{submodel_name[4]}/model.onnx'

    #it is not the openjourney predefined scheduler
    scheduler = PNDMScheduler(beta_start = 0.00085, beta_end = 0.012, beta_schedule = "scaled_linear",
                              num_train_timesteps = 1000)

    prompt = [args.prompt]

    height = 512    #default height of stable Diffusion
    width = 512     #default width of stable Diffusion
    num_inference_steps = args.totalInferences  #number of denoising steps
    guidance_scale = 7.5                        #Scale for classifier-free guidance
    generator = torch.manual_seed(0)            #Seed generator to create the initial latent noise
    batch_size = len(prompt)

    sess_options = ort.SessionOptions()
    poptions_txt_clip = {
        "etglow_greedy": "true",
        "etglow_onnx_shape_params": f'batch={batch_size};sequence={tokenizer.model_max_length}'
    }
    
    poptions_unet = {
        "etglow_greedy": "true",
        "etglow_onnx_shape_params": f'batch={batch_size*2};channels=4;height={height//8};width={width//8};sequence={tokenizer.model_max_length}'
    }
    
    poptions_vae_decode = {
        "etglow_greedy": "true",
        "etglow_onnx_shape_params": f'batch={batch_size};channels=4;height={height//8};width={width//8};channels_image=3;height_image={height};width_image={width}'
    }
    
    session_txt_clip_cpu = ort.InferenceSession(model_txt_clip_path)
    session_txt_clip_et = ort.InferenceSession(model_txt_clip_path, sess_options, providers = ['EtGlowExecutionProvider'], provider_options = [poptions_txt_clip])
    in_name_input_ids = session_txt_clip_cpu.get_inputs()[0].name

    session_unet_cpu = ort.InferenceSession(model_unet_path)
    session_unet_et = ort.InferenceSession(model_unet_path, sess_options, providers = ['EtGlowExecutionProvider'], provider_options = [poptions_unet])
    in_name_sample = session_unet_cpu.get_inputs()[0].name
    in_name_timestep = session_unet_cpu.get_inputs()[1].name
    in_name_encoder_hidden_states = session_unet_cpu.get_inputs()[2].name

    session_vae_decode_cpu = ort.InferenceSession(model_vae_decode_path)
    session_vae_decode_et = ort.InferenceSession(model_vae_decode_path, sess_options, providers = ['EtGlowExecutionProvider'], provider_options = [poptions_vae_decode])
    in_name_latent_sample = session_vae_decode_cpu.get_inputs()[0].name

    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, return_tensors = "pt")
    input_ids = text_input.input_ids.numpy()
    input_ids = input_ids.astype(np.int32)

    if (args.silicon):
        text_embeddings = session_txt_clip_et.run([], {in_name_input_ids: input_ids})[0] 
    else:
        text_embeddings = session_txt_clip_cpu.run([], {in_name_input_ids: input_ids})[0]

    max_length = text_input.input_ids.shape[-1]
    text_uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_input = text_uncond_input.input_ids.numpy()
    uncond_input = uncond_input.astype(np.int32)

    start = time.time()
    if (args.silicon):
        uncond_embeddings = session_txt_clip_et.run([], {in_name_input_ids: uncond_input})[0]
    else:
        uncond_embeddings = session_txt_clip_cpu.run([], {in_name_input_ids: uncond_input})[0]
        
    text_embeddings = torch.cat([torch.from_numpy(uncond_embeddings), torch.from_numpy(text_embeddings)])
    end = time.time()
    diff_time = np.round((end-start)*1000,2)
    print(f'Time to run txt_clip tooks {str(diff_time)}ms)')
    
    latents = torch.randn((batch_size, 4, height // 8, width // 8), generator = generator)

    #initialize th escheduler with the inference steps
    scheduler.set_timesteps(num_inference_steps)

    latents = latents * scheduler.init_noise_sigma

    if (args.datatype == "fp16"):
        text_embeddings = text_embeddings.numpy().astype(np.float16)
    else:
        text_embeddings = text_embeddings.numpy()
        
    for t in tqdm(scheduler.timesteps):
        #expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep = t)

        if (args.datatype == "fp16"):
            latent_model_input = latent_model_input.numpy().astype(np.float16)
        else:
            latent_model_input = latent_model_input.numpy()
            
        #predict the noise residual
        with torch.no_grad():
            start = time.time()
            if (args.silicon):
                noise_pred = session_unet_et.run([], {in_name_sample: latent_model_input,
                                                      in_name_timestep: [np.array(t)],
                                                      in_name_encoder_hidden_states: text_embeddings})[0]
            else:
                noise_pred = session_unet_cpu.run([], {in_name_sample: latent_model_input,
                                                       in_name_timestep: [np.array(t)],
                                                       in_name_encoder_hidden_states: text_embeddings})[0]                
            end = time.time()
            diff_time = np.round((end-start)*1000, 2)
            print(f'Time to run unet tooks {str(diff_time)}ms) at iteration {t}')

        #perform guidance
        noise_pred = torch.from_numpy(noise_pred) #we convert it to tensor, if not chunk does not work\n",
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        #compute the previous noisiy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    #scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents

    if (args.datatype == "fp16"):
        latents = latents.numpy().astype(np.float16)
    else:
        latents = latents.numpy()
        
    with torch.no_grad():
        start = time.time()
        if (args.silicon):
            image = session_vae_decode_et.run([], {in_name_latent_sample: latents})
        else:
            image = session_vae_decode_cpu.run([], {in_name_latent_sample: latents})                       

        end = time.time()
        diff_time = np.round((end-start)*1000,2)
        print(f'Time to run vae_decode tooks {str(diff_time)}ms)')

    if (args.datatype == "fp16"):        
        image = image[0].astype(np.float32)
    else:
        image = image[0]

    image = (torch.from_numpy(image) / 2 + 0.5).clamp(0, 1)
    image = image.permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    pil_images[0].save(f'{prompt[0].replace(" ", "")}_{args.datatype}.png')
    

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
