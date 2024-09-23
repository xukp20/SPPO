#!/bin/bash
export HF_HOME="/cephfs/shared/hf_cache"
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"


# iter_num=3
RM_MODEL_NAME="/cephfs/shared/zhangge/models/general_preference/2b_gemma_it/batch32_tau01_no_sft_1e5_sky80k/"
# set RM_CONFIGS as a json string
RM_CONFIGS="{\"is_general_preference\": true, \"tau\": 0.1}"

RM_MODEL_SUFFIX="gp_2b_tau01"
SUFFIX="_${RM_MODEL_SUFFIX}"
export RM_MODEL_NAME    
export SUFFIX
export RM_CONFIGS

start_iter=1
iter_num=3
for i in $(seq 1 $iter_num); do
    echo "iter $i"
    if [ "$i" -eq 1 ]; then
        MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
    else
        MODEL=$OUTPUT_DIR
    fi
    OUTPUT_DIR="checkpoints/Llama-3-8B-Instruct-SPPO-Iter${i}${SUFFIX}-table"
    PROMPT="UCLA-AGI/data-mistral-7b-instruct-sppo-iter${i}"

    OUT="data-llama-3-8b-instruct-sppo-iter${i}-table${SUFFIX}" 
    DATASET="synthetic_data_llama-3-8b-instruct-sppo-iter${i}-table${SUFFIX}_score"

    if [ "$i" -lt $start_iter ]; then
        continue
    fi
    
    bash scripts/generate_table_gp.sh --model $MODEL --prompt $PROMPT --out_path $OUT
    bash scripts/pipeline_table.sh --model $MODEL --iter $i --dataset $DATASET --output_dir $OUTPUT_DIR --num 1
done