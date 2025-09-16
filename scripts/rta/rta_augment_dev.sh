## switchboard from babylm7.5M
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_7.5M/switchboard.train \
    --output_dir data_augmented/dev \
    --ratio 0.33 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_7.5M/babylm_7.5M.vectors.txt \
    --vocab_file models/glove/babylm_7.5M/babylm_7.5M.vocab.txt \
    --top_k 15 \
    --temperature 0.9 \
    --do_eval \