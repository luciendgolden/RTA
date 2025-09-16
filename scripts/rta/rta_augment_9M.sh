# RTA augmentation on 9M dataset, 1M synthetic (Proportion 90% sample base)
echo "Starting RTA augmentation on 9M dataset, 1M synthetic... r=0.11"

## bnc
echo "Processing BNC Spoken..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_9M/bnc_spoken.train \
    --output_dir data_augmented/babylm_9M \
    --ratio 0.11 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_9M/babylm_9M.vectors.txt \
    --vocab_file models/glove/babylm_9M/babylm_9M.vocab.txt \
    --top_k 10 \
    --temperature 0.8 \
    --do_eval \
    --verbosity 1 > "logs/babylm_9M/bnc_spoken.log" 2>&1 &

## childes
echo "Processing Childes..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_9M/childes.train \
    --output_dir data_augmented/babylm_9M \
    --ratio 0.11 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_9M/babylm_9M.vectors.txt \
    --vocab_file models/glove/babylm_9M/babylm_9M.vocab.txt \
    --top_k 10 \
    --temperature 0.8 \
    --do_eval \
    --verbosity 1 > "logs/babylm_9M/childes.log" 2>&1 &

## gutenberg
echo "Processing Gutenberg..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_9M/gutenberg.train \
    --output_dir data_augmented/babylm_9M \
    --ratio 0.11 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_9M/babylm_9M.vectors.txt \
    --vocab_file models/glove/babylm_9M/babylm_9M.vocab.txt \
    --top_k 10 \
    --temperature 0.8 \
    --do_eval \
    --verbosity 1 > "logs/babylm_9M/gutenberg.log" 2>&1 &

## openSubtitles
echo "Processing OpenSubtitles..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_9M/open_subtitles.train \
    --output_dir data_augmented/babylm_9M \
    --ratio 0.11 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_9M/babylm_9M.vectors.txt \
    --vocab_file models/glove/babylm_9M/babylm_9M.vocab.txt \
    --top_k 10 \
    --temperature 0.8 \
    --do_eval \
    --verbosity 1 > "logs/babylm_9M/open_subtitles.log" 2>&1 &

## simple english wiki
echo "Processing Simple Wiki..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_9M/simple_wiki.train \
    --output_dir data_augmented/babylm_9M \
    --ratio 0.11 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_9M/babylm_9M.vectors.txt \
    --vocab_file models/glove/babylm_9M/babylm_9M.vocab.txt \
    --top_k 10 \
    --temperature 0.8 \
    --do_eval \
    --verbosity 1 > "logs/babylm_9M/simple_wiki.log" 2>&1 &

## switchboard
echo "Processing Switchboard..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_9M/switchboard.train \
    --output_dir data_augmented/babylm_9M \
    --ratio 0.11 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_9M/babylm_9M.vectors.txt \
    --vocab_file models/glove/babylm_9M/babylm_9M.vocab.txt \
    --top_k 10 \
    --temperature 0.8 \
    --do_eval \
    --verbosity 1 > "logs/babylm_9M/switchboard.log" 2>&1 &

wait
echo "Finished 9M dataset."