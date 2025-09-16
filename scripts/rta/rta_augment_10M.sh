# RTA augmentation on 10M dataset, 10M synthetic (Proportion 0% sample base)
echo "Starting RTA augmentation on 10M dataset, 10M synthetic... r=1.0"

## bnc
echo "Processing BNC Spoken..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_10M/bnc_spoken.train \
    --output_dir data_augmented/rta_10M \
    --ratio 1.0 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_10M/babylm_10M.vectors.txt \
    --vocab_file models/glove/babylm_10M/babylm_10M.vocab.txt \
    --top_k 20 \
    --temperature 1.0 \
    --do_eval \
    --verbosity 1 > "logs/rta_10M/bnc_spoken.log" 2>&1 &

## childes
echo "Processing Childes..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_10M/childes.train \
    --output_dir data_augmented/rta_10M \
    --ratio 1.0 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_10M/babylm_10M.vectors.txt \
    --vocab_file models/glove/babylm_10M/babylm_10M.vocab.txt \
    --top_k 20 \
    --temperature 1.0 \
    --do_eval \
    --verbosity 1 > "logs/rta_10M/childes.log" 2>&1 &

## gutenberg
echo "Processing Gutenberg..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_10M/gutenberg.train \
    --output_dir data_augmented/rta_10M \
    --ratio 1.0 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_10M/babylm_10M.vectors.txt \
    --vocab_file models/glove/babylm_10M/babylm_10M.vocab.txt \
    --top_k 20 \
    --temperature 1.0 \
    --do_eval \
    --verbosity 1 > "logs/rta_10M/gutenberg.log" 2>&1 &

## openSubtitles
echo "Processing OpenSubtitles..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_10M/open_subtitles.train \
    --output_dir data_augmented/rta_10M \
    --ratio 1.0 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_10M/babylm_10M.vectors.txt \
    --vocab_file models/glove/babylm_10M/babylm_10M.vocab.txt \
    --top_k 20 \
    --temperature 1.0 \
    --do_eval \
    --verbosity 1 > "logs/rta_10M/open_subtitles.log" 2>&1 &

## simple english wiki
echo "Processing Simple Wiki..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_10M/simple_wiki.train \
    --output_dir data_augmented/rta_10M \
    --ratio 1.0 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_10M/babylm_10M.vectors.txt \
    --vocab_file models/glove/babylm_10M/babylm_10M.vocab.txt \
    --top_k 20 \
    --temperature 1.0 \
    --do_eval \
    --verbosity 1 > "logs/rta_10M/simple_wiki.log" 2>&1 &

## switchboard
echo "Processing Switchboard..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_10M/switchboard.train \
    --output_dir data_augmented/rta_10M \
    --ratio 1.0 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_10M/babylm_10M.vectors.txt \
    --vocab_file models/glove/babylm_10M/babylm_10M.vocab.txt \
    --top_k 20 \
    --temperature 1.0 \
    --do_eval \
    --verbosity 1 > "logs/rta_10M/switchboard.log" 2>&1 &

wait
echo "Finished 10M dataset."