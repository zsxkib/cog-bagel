# Data prepration

We provide data examples for **T2I**, **Editing**, and **VLM** tasks. The T2I dataset is generated using [FLUX.1‑dev](https://huggingface.co/black-forest-labs/FLUX.1-dev); the editing examples are randomly sampled from [SEED‑Data‑Edit‑Part3](https://huggingface.co/datasets/AILab-CVC/SEED-Data-Edit-Part2-3); and the VLM set is sourced from [LLaVA‑OneVision‑Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data).

We offer examples in both raw-image folder and parquet shard formats. For other data formats, you can use our dataset code as a template and extend it as needed.


1. **Download the sample dataset**

   ```bash
   wget -O bagel_example.zip \
     https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/bagel_example.zip
   unzip bagel_example.zip -d /data
   ```
2. **Expected hierarchy**

   ```text
   bagel_example
   ├── t2i/                           # text-to-image (parquet)
   ├── editing/                       # image editing (parquet)
   │   ├── seedxedit_multi/
   │   └── parquet_info/
   └── vlm/
       ├── images/                    # JPEG / PNG frames
       └── llava_ov_si.jsonl          # vision‑language SFT conversations
   ```
3. Edit every `your_data_path` placeholder in **`data/dataset_info.py`**.
4. *(Optional)*  Extend `DATASET_INFO` with your own parquet shards or JSONL files to mix extra data.

---

# Training

The baseline full‑feature recipe looks like this (replace environment variables with real paths or values):

```shell
torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=8 \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --llm_path $llm_path \
  --vae_path $vae_path \
  --vit_path $vit_path \
  --use_flex True \
  --resume_from $resume_from \
  --results_dir $output_path \
  --checkpoint_dir $ckpt_path \
  --max_latent_size 64  # 32 for low-resolution pre-training
```

- **When fine-tuning BAGEL, please set `max_latent_size=64` to ensure the correct pretrained weights are loaded.**
- For T2I-only fine-tuning, set `visual_und=False`; for VLM-only, set `visual_gen=False`.

 You are encouraged to adjust any of these hyperparameters to fit your GPU budget and the scale of your dataset. If you encounter any issues, please open an issue for assistance. 🎉


## Model config


| Argument                     | Default                                     | Description                                                     |
| ---------------------------- | ------------------------------------------- | --------------------------------------------------------------- |
| `llm_path`                   | `hf/Qwen2.5-0.5B-Instruct`                  | Language‑model backbone (HuggingFace repo or local folder).     |
| `vae_path`                   | `flux/vae/ae.safetensors`                   | Pre‑trained VAE checkpoint for latent diffusion.                |
| `vit_path`                   | `hf/siglip-so400m-14-980-flash-attn2-navit` | SigLIP ViT used for image understanding.                        |
| `max_latent_size`            | `32`                                        | Maximum latent grid side; defines highest generable resolution. |
| `latent_patch_size`          | `2`                                         | VAE pixels represented by one latent patch.                     |
| `vit_max_num_patch_per_side` | `70`                                        | Max ViT patches per image side after resizing.                  |
| `text_cond_dropout_prob`     | `0.1`                                       | Probability to drop text conditioning while training.           |
| `vae_cond_dropout_prob`      | `0.3`                                       | Dropout on VAE latent inputs.                                   |
| `vit_cond_dropout_prob`      | `0.3`                                       | Dropout on visual features.                                     |

*(See `ModelArguments` for many more options.)*


## Data config


| Argument                    | Default                     | Description                                               |
| --------------------------- | --------------------------- | --------------------------------------------------------- |
| `dataset_config_file`       | `data/configs/example.yaml` | YAML that groups datasets and assigns sampling weights.   |
| `num_workers`               | `4`                         | Background workers per rank for the PyTorch `DataLoader`. |
| `prefetch_factor`           | `2`                         | Batches pre‑fetched by each worker.                       |
| `max_num_tokens_per_sample` | `16384`                     | Skip raw samples longer than this.                        |
| `max_num_tokens`            | `36864`                     | Hard cap for a packed batch (prevents OOM).               |
| `max_buffer_size`           | `50`                        | Overflow buffer length for oversized samples.             |
| `data_seed`                 | `42`                        | Seed for reproducible shuffling and sampling.             |


## Training config

| Argument                               | Default                | Description                                            |
| -------------------------------------- | ---------------------- | ------------------------------------------------------ |
| `total_steps`                          | `500_000`              | Optimiser steps to run.                                |
| `lr`                                   | `1e-4`                 | Peak learning rate after warm‑up.                      |
| `lr_scheduler`                         | `constant`             | Learning‑rate schedule (`constant` or `cosine`).       |
| `warmup_steps`                         | `2000`                 | Linear warm‑up duration.                               |
| `ema`                                  | `0.9999`               | Exponential moving‑average decay for model weights.    |
| `max_grad_norm`                        | `1.0`                  | Gradient‑clipping threshold.                           |
| `save_every`                           | `2000`                 | Checkpoint frequency (steps).                          |
| `visual_gen / visual_und`              | `True`                 | Enable image generation / understanding branches.      |
| `freeze_llm / freeze_vit / freeze_vae` | `False / False / True` | Freeze selected modules to save VRAM or for ablations. |
| `use_flex`                             | `True` (in example)    | Enable FLEX packing for higher GPU utilisation.        |
| `sharding_strategy`                    | `HYBRID_SHARD`         | FSDP sharding mode.                                    |
| `num_shard`                            | `8`                    | Parameter shards per rank in HYBRID mode.              |

**Distributed‑launch environment variables**

| Var                           | Meaning                           |
| ----------------------------- | --------------------------------- |
| `num_nodes` / `node_rank`     | Multi‑node orchestration indices. |
| `nproc_per_node`              | Number of GPUs per node.          |
| `master_addr` / `master_port` | NCCL rendezvous endpoint.         |


## Logging config


| Argument         | Default               | Description                                          |
| ---------------- | --------------------- | ---------------------------------------------------- |
| `results_dir`    | `results`             | Root directory for logs and metrics.                 |
| `checkpoint_dir` | `results/checkpoints` | Checkpoints are saved here.                          |
| `log_every`      | `10`                  | Steps between console / W\&B logs.                   |
| `wandb_project`  | `bagel`               | Weights & Biases project name.                       |
| `wandb_name`     | `run`                 | Run name inside the project.                         |
| `wandb_offline`  | `False`               | Switch to offline mode (logs locally, sync later).   |
| `wandb_resume`   | `allow`               | Resumption policy if an existing run ID is detected. |

> **Tip**  Export `WANDB_API_KEY` before launching if you want online dashboards.