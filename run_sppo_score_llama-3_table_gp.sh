#!/bin/bash
export HF_HOME="/cephfs/shared/hf_cache"
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"


# iter_num=3
if [ -z $beta ]; then
    beta=0.001
fi
if [ -z $lr ]; then
    lr=5e-7
fi
if [ -z $clamp_thres ]; then
    clamp_thres=1000
fi

echo "beta: $beta"
echo "lr: $lr"
echo "clamp_thres: $clamp_thres"

export BETA=$beta
export CLAMP_THRES=$clamp_thres

# RM_MODEL_NAME="/cephfs/shared/zhangge/models/general_preference/2b_gemma_it/batch32_tau01_no_sft_1e5_sky80k/"
RM_MODEL_NAME="/cephfs/shared/zhangge/models/general_preference/2b_gemma_it/batch32_tau01_no_sft_1e5_sky80k_epoch2_vh8_w_moe_w_l2"

# RM_MODEL_NAME="/cephfs/shared/zhangge/models/general_preference/8b_llama31/batch32_tau01_no_sft_2e6_sky80k_epoch2"
RM_MODEL_NAME="/cephfs/shared/zhangge/models/general_preference/8b_llama31/batch32_tau01_no_sft_2e6_sky80k_epoch2_vh6_w_moe_w_l2"

# set RM_CONFIGS as a json string
# RM_CONFIGS="{\"is_general_preference\": true, \"tau\": 0.1, \"value_head_dim\": 8}"
RM_CONFIGS="{\"is_general_preference\": true, \"tau\": 0.1, \"value_head_dim\": 6}"

RM_MODEL_SUFFIX="gp_8b"
LR_SUFFIX=""
if [ "$lr" != "5e-7" ]; then
    LR_SUFFIX="_${lr}"
fi

# RM_MODEL_SUFFIX="gp_2b_tau01"
SUFFIX="_${RM_MODEL_SUFFIX}${LR_SUFFIX}"
export RM_MODEL_NAME    
export SUFFIX
export RM_CONFIGS

start_iter=1
iter_num=3
for i in $(seq 1 $iter_num); do
    if [ "$i" -eq 1 ]; then
        MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
    else
        MODEL=$OUTPUT_DIR
    fi
    OUTPUT_DIR="checkpoints/Llama-3-8B-Instruct-SPPO-score-Iter${i}${SUFFIX}-table-${beta}"
    PROMPT="UCLA-AGI/data-mistral-7b-instruct-sppo-iter${i}"

    OUT="data-llama-3-8b-instruct-sppo-score-iter${i}${SUFFIX}-table-${beta}" 
    DATASET="synthetic_data_llama-3-8b-instruct-sppo-score-iter${i}${SUFFIX}-table-${beta}_score"

    if [ "$i" -lt $start_iter ]; then
        continue
    fi
    
    bash scripts/generate_score_table_gp.sh --model $MODEL --prompt $PROMPT --out_path $OUT
    bash scripts/pipeline_score_table.sh --model $MODEL --iter $i --dataset $DATASET --output_dir $OUTPUT_DIR --num 1 --beta $beta --learning_rate $lr
done
