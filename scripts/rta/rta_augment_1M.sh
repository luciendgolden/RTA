# RTA augmentation on 1M dataset, 9M synthetic (Proportion 10% sample base)
echo "Starting RTA augmentation on 1M dataset, 9M synthetic... r=9.0"

## bnc
echo "Processing BNC Spoken..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_1M/bnc_spoken.train \
    --output_dir data_augmented/babylm_1M \
    --ratio 9.0 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_1M/babylm_1M.vectors.txt \
    --vocab_file models/glove/babylm_1M/babylm_1M.vocab.txt \
    --top_k 50 \
    --temperature 1.3 \
    --do_eval \
    --verbosity 1 > "logs/babylm_1M/bnc_spoken.log" 2>&1 &

## childes
echo "Processing CHILDES..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_1M/childes.train \
    --output_dir data_augmented/babylm_1M \
    --ratio 9.0 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_1M/babylm_1M.vectors.txt \
    --vocab_file models/glove/babylm_1M/babylm_1M.vocab.txt \
    --top_k 50 \
    --temperature 1.3 \
    --do_eval \
    --verbosity 1 > "logs/babylm_1M/childes.log" 2>&1 &

## gutenberg
echo "Processing Gutenberg..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_1M/gutenberg.train \
    --output_dir data_augmented/babylm_1M \
    --ratio 9.0 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_1M/babylm_1M.vectors.txt \
    --vocab_file models/glove/babylm_1M/babylm_1M.vocab.txt \
    --top_k 50 \
    --temperature 1.3 \
    --do_eval \
    --verbosity 1 > "logs/babylm_1M/gutenberg.log" 2>&1 &

## openSubtitles
echo "Processing OpenSubtitles..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_1M/open_subtitles.train \
    --output_dir data_augmented/babylm_1M \
    --ratio 9.0 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_1M/babylm_1M.vectors.txt \
    --vocab_file models/glove/babylm_1M/babylm_1M.vocab.txt \
    --top_k 50 \
    --temperature 1.3 \
    --do_eval \
    --verbosity 1 > "logs/babylm_1M/open_subtitles.log" 2>&1 &

## simple english wiki
echo "Processing Simple English Wiki..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_1M/simple_wiki.train \
    --output_dir data_augmented/babylm_1M \
    --ratio 9.0 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_1M/babylm_1M.vectors.txt \
    --vocab_file models/glove/babylm_1M/babylm_1M.vocab.txt \
    --top_k 50 \
    --temperature 1.3 \
    --do_eval \
    --verbosity 1 > "logs/babylm_1M/simple_wiki.log" 2>&1 &

## switchboard
echo "Processing Switchboard..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_1M/switchboard.train \
    --output_dir data_augmented/babylm_1M \
    --ratio 9.0 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_1M/babylm_1M.vectors.txt \
    --vocab_file models/glove/babylm_1M/babylm_1M.vocab.txt \
    --top_k 50 \
    --temperature 1.3 \
    --do_eval \
    --verbosity 1 > "logs/babylm_1M/switchboard.log" 2>&1 &

wait
echo "Finished RTA augmentation on 1M dataset, 9M synthetic... r=9.0"