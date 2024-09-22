export HF_ENDPOINT=https://hf-mirror.com
MODEL="/cephfs/shared/hf_cache/hub/models--google--gemma-2b-it/snapshots/1027d96c1638a27f01ae935cd98bac7d1a01686c"
OUTDIR="test"

PAIRS=5
FRAC=0
PROMPTS="UCLA-AGI/data-mistral-7b-instruct-sppo-iter1"

python3 rank_pairrm_llama.py --model $MODEL --output_dir $OUTDIR --numgpu 1 --pairs $PAIRS