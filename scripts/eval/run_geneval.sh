# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

GPUS=8

# generate images
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/gen/gen_images_mp.py \
    --output_dir $output_path/images \
    --metadata_file ./eval/gen/geneval/prompts/evaluation_metadata_long.jsonl \
    --batch_size 1 \
    --num_images 4 \
    --resolution 1024 \
    --max_latent_size 64 \
    --model-path $model_path \
    # --metadata_file ./eval/gen/geneval/prompts/evaluation_metadata.jsonl \


# calculate score
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/gen/geneval/evaluation/evaluate_images_mp.py \
    $OUTPUT_DIR/images \
    --outfile $OUTPUT_DIR/results.jsonl \
    --model-path ./eval/gen/geneval/model


# summarize score
python ./eval/gen/geneval/evaluation/summary_scores.py $OUTPUT_DIR/results.jsonl