# RTA augmentation on 7.5M dataset, 2.5M synthetic (Proportion 75% sample base)
echo "Starting RTA augmentation on base7.5M dataset, 2.5M synthetic... r=0.33"

## bnc
echo "Processing BNC Spoken..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_7.5M/bnc_spoken.train \
    --output_dir data_augmented/babylm_7.5M \
    --ratio 0.33 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_7.5M/babylm_7.5M.vectors.txt \
    --vocab_file models/glove/babylm_7.5M/babylm_7.5M.vocab.txt \
    --top_k 15 \
    --temperature 0.9 \
    --do_eval \
    --verbosity 1 > "logs/babylm_7.5M/bnc_spoken.log" 2>&1 &

## childes
echo "Processing Childes..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_7.5M/childes.train \
    --output_dir data_augmented/babylm_7.5M \
    --ratio 0.33 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_7.5M/babylm_7.5M.vectors.txt \
    --vocab_file models/glove/babylm_7.5M/babylm_7.5M.vocab.txt \
    --top_k 15 \
    --temperature 0.9 \
    --do_eval \
    --verbosity 1 > "logs/babylm_7.5M/childes.log" 2>&1 &

## gutenberg
echo "Processing Gutenberg..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_7.5M/gutenberg.train \
    --output_dir data_augmented/babylm_7.5M \
    --ratio 0.33 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_7.5M/babylm_7.5M.vectors.txt \
    --vocab_file models/glove/babylm_7.5M/babylm_7.5M.vocab.txt \
    --top_k 15 \
    --temperature 0.9 \
    --do_eval \
    --verbosity 1 > "logs/babylm_7.5M/gutenberg.log" 2>&1 &

## openSubtitles
echo "Processing OpenSubtitles..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_7.5M/open_subtitles.train \
    --output_dir data_augmented/babylm_7.5M \
    --ratio 0.33 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_7.5M/babylm_7.5M.vectors.txt \
    --vocab_file models/glove/babylm_7.5M/babylm_7.5M.vocab.txt \
    --top_k 15 \
    --temperature 0.9 \
    --do_eval \
    --verbosity 1 > "logs/babylm_7.5M/open_subtitles.log" 2>&1 &

## simple wiki
echo "Processing Simple Wiki..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_7.5M/simple_wiki.train \
    --output_dir data_augmented/babylm_7.5M \
    --ratio 0.33 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_7.5M/babylm_7.5M.vectors.txt \
    --vocab_file models/glove/babylm_7.5M/babylm_7.5M.vocab.txt \
    --top_k 15 \
    --temperature 0.9 \
    --do_eval \
    --verbosity 1 > "logs/babylm_7.5M/simple_wiki.log" 2>&1 &

## switchboard
echo "Processing Switchboard..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_7.5M/switchboard.train \
    --output_dir data_augmented/babylm_7.5M \
    --ratio 0.33 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_7.5M/babylm_7.5M.vectors.txt \
    --vocab_file models/glove/babylm_7.5M/babylm_7.5M.vocab.txt \
    --top_k 15 \
    --temperature 0.9 \
    --do_eval \
    --verbosity 1 > "logs/babylm_7.5M/switchboard.log" 2>&1 &

wait
echo "Finished 7.5M dataset."