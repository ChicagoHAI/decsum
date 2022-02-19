#!/bin/bash
OUTPUT_DIR=data/out/
YELP_OUTPUT_DIR=data/test_decsum/ # output dir of preprocessing step
DATA_DIR=${YELP_OUTPUT_DIR}/50reviews/
CACHE_DIR=data/test_decsum/transformers_cache/

python -m models.transformers.main \
    --max_epochs 3 \
    --max_seq_length 30 \
    --output_dir $OUTPUT_DIR \
    --data_dir $DATA_DIR \
    --model_name_or_path allenai/longformer-base-4096 \
    --warmup_steps 500 \
    --do_train \
    --cache_dir $CACHE_DIR \
    --fp16