# ByteDance-Seed's BAGEL: Unified Multimodal AI in One Model

[![Replicate](https://replicate.com/zsxkib/bagel/badge)](https://replicate.com/zsxkib/bagel)

This repository contains a Cog implementation of **BAGEL (Unified Model for Multimodal Understanding and Generation)**, ByteDance's incredible 7B parameter multimodal model that can generate images, edit images, AND understand images—all in one unified model. This is the kind of AI we've been waiting for: one model that truly does it all.

BAGEL uses a Mixture-of-Transformer-Experts (MoT) architecture and can even do chain-of-thought reasoning, meaning it can "think" through complex tasks before generating its output. Whether you want to create stunning images from text, edit existing photos with natural language instructions, or have deep conversations about what's happening in an image, BAGEL handles it all seamlessly.

**What makes BAGEL special:**
- **One model, three superpowers**: text-to-image generation, image editing, and image understanding
- **Chain-of-thought reasoning**: The model can think through tasks step-by-step for better results
- **Emerging capabilities**: As ByteDance scaled up training, the model developed increasingly sophisticated abilities
- **Production-ready**: 7B active parameters (14B total) that outperform many specialized models

**Model links and information:**
*   Original Project: [ByteDance-Seed/Bagel](https://github.com/ByteDance-Seed/Bagel)
*   Research Paper: [Emerging Properties in Unified Multimodal Pretraining](https://arxiv.org/abs/2505.14683)
*   Model Weights: [ByteDance-Seed/BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)
*   Project Website: [bagel-ai.org](https://bagel-ai.org/)
*   Demo: [demo.bagel-ai.org](https://demo.bagel-ai.org/)
*   This Cog packaging by: [zsxkib on GitHub](https://github.com/zsxkib) / [@zsakib_ on Twitter](https://twitter.com/zsakib_)

## Prerequisites

*   **Docker**: You'll need Docker to build and run the Cog container. [Install Docker](https://docs.docker.com/get-docker/).
*   **Cog**: Cog is required to build and run this model locally. [Install Cog](https://github.com/replicate/cog#install).
*   **NVIDIA GPU**: You'll need a powerful NVIDIA GPU with at least 40GB of memory (A100 or H100 recommended) to run the full model.

## Run locally: It just works!

Running BAGEL with Cog is incredibly straightforward. The beauty of this setup is that everything is handled automatically—model downloads, environment setup, dependencies—you just run a command and watch the magic happen.

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/zsxkib/cog-bagel.git
    cd cog-bagel
    ```

2.  **Run the model:**
    The first time you run any command, Cog will download the 7B+ model weights from Hugging Face (this takes a few minutes), but after that, everything runs lightning fast.

    **Text-to-Image Generation:**
    ```bash
    # Generate a beautiful landscape
    cog predict -i prompt="A serene mountain landscape at sunset with golden light reflecting on a peaceful lake"
    
    # Try with chain-of-thought reasoning for complex scenes
    cog predict -i prompt="A futuristic city floating in the clouds" -i enable_thinking=true
    
    # Get creative with the prompt
    cog predict -i prompt="A cat wearing a astronaut helmet, digital art, trending on artstation"
    ```

    **Image Editing (This is where BAGEL really shines!):**
    ```bash
    # Edit any image with natural language
    cog predict \
      -i prompt="Change the sky to a starry night" \
      -i image=@test_images/landscape.jpg \
      -i task="image-editing"
    
    # More dramatic edits
    cog predict \
      -i prompt="Make it look like it's underwater with fish swimming around" \
      -i image=@your_photo.jpg \
      -i task="image-editing" \
      -i cfg_img_scale=2.0
    
    # Smart editing with reasoning
    cog predict \
      -i prompt="Transform this into a vintage 1920s photograph" \
      -i image=@modern_photo.jpg \
      -i task="image-editing" \
      -i enable_thinking=true
    ```

    **Image Understanding:**
    ```bash
    # Ask questions about any image
    cog predict \
      -i prompt="What's happening in this image? Describe it in detail." \
      -i image=@complex_scene.jpg \
      -i task="image-understanding"
    
    # Get deep analysis with reasoning
    cog predict \
      -i prompt="Analyze the artistic style and composition of this painting" \
      -i image=@artwork.jpg \
      -i task="image-understanding" \
      -i enable_thinking=true
    
    # Even understand memes!
    cog predict \
      -i prompt="Can someone explain what's funny about this meme?" \
      -i image=@test_images/meme.jpg \
      -i task="image-understanding" \
      -i enable_thinking=true
    ```

    **Advanced Usage:**
    ```bash
    # Fine-tune generation parameters
    cog predict \
      -i prompt="A cyberpunk robot in neon city" \
      -i cfg_text_scale=6.0 \
      -i num_inference_steps=30 \
      -i seed=42 \
      -i output_format="png"
    
    # Careful image editing with specific controls
    cog predict \
      -i prompt="Add gentle snow falling" \
      -i image=@winter_scene.jpg \
      -i task="image-editing" \
      -i cfg_img_scale=1.8 \
      -i cfg_renorm_type="text_channel" \
      -i timestep_shift=2.5
    ```

## How it works

This Cog implementation faithfully follows the original BAGEL research and codebase. Here's what happens under the hood:

*   **`setup()` method**: When the container starts up:
    1.  Downloads the complete BAGEL-7B-MoT model from Replicate's high-speed CDN (~28GB)
    2.  Sets up the Mixture-of-Transformer-Experts architecture with proper device mapping
    3.  Initializes the dual encoders (VAE for pixel-level features, ViT for semantic features)
    4.  Prepares the unified tokenizer and special tokens for multimodal processing
    5.  Loads everything onto your GPU with optimized memory management

*   **`predict()` method**: The magic happens here:
    1.  **Task Detection**: Automatically configures parameters based on whether you're doing text-to-image, editing, or understanding
    2.  **Chain-of-Thought**: If enabled, the model first "thinks" through the problem before generating output
    3.  **Multimodal Processing**: Uses both VAE and ViT encoders to understand images at both pixel and semantic levels
    4.  **Unified Generation**: The MoT architecture seamlessly switches between generating text tokens and image tokens
    5.  **Smart Inference**: Applies different CFG (Classifier-Free Guidance) strategies optimized for each task type

The beauty of BAGEL is that it's not three separate models stitched together—it's one unified model that learned to do all these tasks simultaneously, leading to emergent capabilities that specialized models can't achieve.

## Why BAGEL is a game-changer

Traditional approaches require separate models for each task: DALL-E for text-to-image, InstructPix2Pix for editing, GPT-4V for understanding. BAGEL does all of this in a single 7B parameter model that often outperforms these specialized solutions.

The research shows that as ByteDance scaled up training, BAGEL developed increasingly sophisticated capabilities—first basic understanding and generation, then editing, and finally complex intelligent editing that requires deep visual reasoning. This is exactly the kind of emergent behavior we've been hoping to see in multimodal AI.

## Deploy to Replicate

Want to share your BAGEL model with the world? Push it to Replicate:

```bash
cog login
cog push r8.im/your-username/bagel
```

## License

This implementation follows the original BAGEL project's Apache 2.0 license. The BAGEL model and research are from ByteDance-Seed.

---

⭐ Star this on [GitHub](https://github.com/zsxkib/cog-bagel)!

👋 Follow `zsxkib` on [Twitter/X](https://twitter.com/zsakib_)

**Enjoying BAGEL?** Check out the original project and give the ByteDance team some love: [github.com/ByteDance-Seed/Bagel](https://github.com/ByteDance-Seed/Bagel)
