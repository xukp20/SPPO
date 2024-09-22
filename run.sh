for b in 0.003 0.03; do
    export beta=$b
    bash run_sppo_score_llama-3_table.sh
done