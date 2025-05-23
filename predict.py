# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
import random
import numpy as np
from typing import Optional

import torch
from PIL import Image
from cog import BasePredictor, Input, Path, BaseModel

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer

# Model cache configuration
MODEL_CACHE = "model_cache"
BASE_URL = "https://weights.replicate.delivery/default/bytedance-bagel/model_cache/"

# Set up environment variables for model caching
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

# Add WebP MIME type support
import mimetypes
mimetypes.add_type("image/webp", ".webp")


class BagelOutput(BaseModel):
    """Output from BAGEL model inference"""
    image: Optional[Path] = None
    text: Optional[str] = None
    

def download_weights(url: str, dest: str) -> None:
    """Download model weights using pget with parallel downloading"""
    start = time.time()
    print(f"[+] Downloading from: {url}")
    print(f"[+] Destination: {dest}")
    
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    
    try:
        print(f"[+] Running: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
        print(f"[+] Download completed in {time.time() - start:.1f}s")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Download failed with exit code {e.returncode}")
        raise


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the BAGEL model into memory to make running multiple predictions efficient"""
        print("[+] Setting up BAGEL-7B-MoT model...")
        
        # Create model cache directory
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        # Download BAGEL model weights from CDN
        print("[+] Downloading BAGEL-7B-MoT model weights...")
        model_files = [
            "BAGEL-7B-MoT.tar",
            "models--ByteDance-Seed--BAGEL-7B-MoT.tar",
        ]

        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            
            # Only download if extracted directory doesn't exist
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        print("[+] Model weights download completed")
        
        # Set up model path
        model_path = os.path.join(MODEL_CACHE, "BAGEL-7B-MoT")

        # Initialize model configurations
        print("[+] Loading model configurations...")
        
        # LLM config
        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        # ViT config
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        # VAE loading
        vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

        # BAGEL config
        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config, 
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
            latent_patch_size=2,
            max_latent_size=64,
        )

        # Initialize model architecture
        print("[+] Initializing model architecture...")
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        # Prepare tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        # Prepare image transforms
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        # Set up device mapping for GPU inference
        print("[+] Setting up GPU device mapping...")
        max_mem_per_gpu = "80GiB"  # For A100/H100
        
        device_map = infer_auto_device_map(
            model,
            max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )

        # Ensure related modules are on the same device
        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]

        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for module in same_device_modules:
                if module in device_map:
                    device_map[module] = first_device
                else:
                    device_map[module] = "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            for module in same_device_modules:
                if module in device_map:
                    device_map[module] = first_device

        print(f"[+] Device mapping: {device_map}")
        
        # Load model checkpoint
        print("[+] Loading model checkpoint...")
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder="/tmp/offload"
        )

        model = model.eval()
        
        # Create inferencer
        self.inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids
        )
        
        print("[+] BAGEL model setup completed successfully!")

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt for generation, editing, or understanding"
        ),
        image: Path = Input(
            description="Input image for editing or understanding tasks",
            default=None
        ),
        task: str = Input(
            description="Task to perform",
            choices=["text-to-image", "image-editing", "image-understanding"],
            default="text-to-image"
        ),
        enable_thinking: bool = Input(
            description="Enable chain-of-thought reasoning for better results",
            default=False
        ),
        
        # Generation parameters
        cfg_text_scale: float = Input(
            description="Text guidance scale for how closely to follow the prompt",
            ge=1.0,
            le=20.0,
            default=4.0,
        ),
        cfg_img_scale: float = Input(
            description="Image guidance scale for preserving input image details",
            ge=1.0,
            le=10.0,
            default=1.5,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            ge=1,
            le=100,
            default=50
        ),
        timestep_shift: float = Input(
            description="Distribution of denoising steps between composition and details",
            ge=1.0,
            le=10.0,
            default=3.0,
        ),
        
        # Advanced parameters
        cfg_renorm_type: str = Input(
            description="CFG renormalization method",
            choices=["global", "local", "text_channel"],
            default="global"
        ),
        cfg_renorm_min: float = Input(
            description="Minimum CFG renorm value",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
        
        # Output parameters
        seed: Optional[int] = Input(
            description="Random seed for reproducible results",
            default=None,
        ),
        output_format: str = Input(
            description="Output image format",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Image compression quality for lossy formats",
            ge=1,
            le=100,
            default=90,
        ),
    ) -> BagelOutput:
        """
        Unified BAGEL inference for text-to-image generation, image editing, and image understanding.
        
        BAGEL is a 7B parameter multimodal model that can generate both text and images with 
        optional chain-of-thought reasoning for improved results.
        """
        
        # Set random seed for reproducibility
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"[+] Using seed: {seed}")
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Validate inputs and load image if provided
        input_image = None
        if image is not None:
            input_image = Image.open(str(image)).convert("RGB")
            print(f"[+] Loaded input image: {input_image.size}")

        # Validate task requirements
        if task in ["image-editing", "image-understanding"] and input_image is None:
            raise ValueError(f"Task '{task}' requires an input image")

        # Configure inference parameters based on task
        if task == "text-to-image":
            print("[+] Running text-to-image generation")
            inference_params = {
                "understanding_output": False,
                "think": enable_thinking,
                "cfg_text_scale": cfg_text_scale,
                "cfg_img_scale": 1.0,  # No image guidance for T2I
                "cfg_interval": [0.4, 1.0],
                "cfg_renorm_type": "global",  # Best for T2I
                "cfg_renorm_min": 1.0,  # Disable renorm for T2I
                "timestep_shift": timestep_shift,
                "num_timesteps": num_inference_steps,
                "max_think_token_n": 1000,
                "do_sample": False,
                "text_temperature": 0.3,
            }
            
        elif task == "image-editing":
            print("[+] Running image editing")
            inference_params = {
                "understanding_output": False,
                "think": enable_thinking,
                "cfg_text_scale": cfg_text_scale,
                "cfg_img_scale": cfg_img_scale,
                "cfg_interval": [0.0, 1.0],  # Full interval for editing
                "cfg_renorm_type": cfg_renorm_type,
                "cfg_renorm_min": cfg_renorm_min,
                "timestep_shift": timestep_shift,
                "num_timesteps": num_inference_steps,
                "max_think_token_n": 1000,
                "do_sample": False,
                "text_temperature": 0.3,
            }
            
        elif task == "image-understanding":
            print("[+] Running image understanding")
            inference_params = {
                "understanding_output": True,
                "think": enable_thinking,
                "max_think_token_n": 1000,
                "do_sample": False,
                "text_temperature": 0.3,
            }
        
        # Run BAGEL inference
        print(f"[+] Processing prompt: {prompt}")
        output_dict = self.inferencer(
            image=input_image,
            text=prompt,
            **inference_params
        )
        
        # Handle output based on task type
        if task == "image-understanding":
            # Return text output for understanding tasks
            output_text = output_dict.get('text', 'No response generated')
            
            # Format chain-of-thought output if enabled
            if enable_thinking and '<think>' in output_text:
                parts = output_text.split('<think>')
                if len(parts) > 1:
                    thinking_part = parts[1].split('</think>')[0] if '</think>' in parts[1] else parts[1]
                    answer_part = parts[1].split('</think>')[1] if '</think>' in parts[1] else ""
                    output_text = f"Reasoning: {thinking_part.strip()}\n\nAnswer: {answer_part.strip()}"
            
            print(f"[+] Generated text response ({len(output_text)} characters)")
            return BagelOutput(text=output_text)
        
        else:
            # Return image output for generation/editing tasks
            output_image = output_dict.get('image')
            if output_image is None:
                raise RuntimeError("No image was generated")
            
            # Log thinking process if enabled
            thinking_text = output_dict.get('text')
            if thinking_text:
                print(f"[+] Chain-of-thought: {thinking_text}")
            
            # Ensure image is in RGB mode
            if output_image.mode != "RGB":
                output_image = output_image.convert("RGB")

            # Prepare image saving parameters
            extension = output_format.lower()
            save_params = {}

            if output_format != "png":
                save_params["quality"] = output_quality
                save_params["optimize"] = True

            if extension == "jpg":
                extension = "jpeg"

            # Save image and return path
            output_path = Path(f"output.{extension}")
            output_image.save(str(output_path), **save_params)
            print(f"[+] Generated {output_image.size[0]}x{output_image.size[1]} image saved as {output_format.upper()}")

            return BagelOutput(image=output_path)
