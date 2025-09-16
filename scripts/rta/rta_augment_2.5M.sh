# RTA augmentation on 2.5M dataset, 7.5M synthetic (Proportion 25% sample base)
echo "Starting RTA augmentation on 2.5M dataset, 7.5M synthetic... r=3.0"

## bnc
echo "Processing BNC Spoken..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_2.5M/bnc_spoken.train \
    --output_dir data_augmented/babylm_2.5M \
    --ratio 3.0 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_2.5M/babylm_2.5M.vectors.txt \
    --vocab_file models/glove/babylm_2.5M/babylm_2.5M.vocab.txt \
    --top_k 30 \
    --temperature 1.1 \
    --do_eval \
    --verbosity 1 > "logs/babylm_2.5M/bnc_spoken.log" 2>&1 &

## childes
echo "Processing CHILDES..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_2.5M/childes.train \
    --output_dir data_augmented/babylm_2.5M \
    --ratio 3.0 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_2.5M/babylm_2.5M.vectors.txt \
    --vocab_file models/glove/babylm_2.5M/babylm_2.5M.vocab.txt \
    --top_k 30 \
    --temperature 1.1 \
    --do_eval \
    --verbosity 1 > "logs/babylm_2.5M/childes.log" 2>&1 &

## gutenberg
echo "Processing Gutenberg..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_2.5M/gutenberg.train \
    --output_dir data_augmented/babylm_2.5M \
    --ratio 3.0 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_2.5M/babylm_2.5M.vectors.txt \
    --vocab_file models/glove/babylm_2.5M/babylm_2.5M.vocab.txt \
    --top_k 30 \
    --temperature 1.1 \
    --do_eval \
    --verbosity 1 > "logs/babylm_2.5M/gutenberg.log" 2>&1 &

## openSubtitles
echo "Processing OpenSubtitles..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_2.5M/open_subtitles.train \
    --output_dir data_augmented/babylm_2.5M \
    --ratio 3.0 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_2.5M/babylm_2.5M.vectors.txt \
    --vocab_file models/glove/babylm_2.5M/babylm_2.5M.vocab.txt \
    --top_k 30 \
    --temperature 1.1 \
    --do_eval \
    --verbosity 1 > "logs/babylm_2.5M/open_subtitles.log" 2>&1 &

## simple english wiki
echo "Processing Simple English Wiki..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_2.5M/simple_wiki.train \
    --output_dir data_augmented/babylm_2.5M \
    --ratio 3.0 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_2.5M/babylm_2.5M.vectors.txt \
    --vocab_file models/glove/babylm_2.5M/babylm_2.5M.vocab.txt \
    --top_k 30 \
    --temperature 1.1 \
    --do_eval \
    --verbosity 1 > "logs/babylm_2.5M/simple_wiki.log" 2>&1 &

## switchboard
echo "Processing Switchboard..."
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_2.5M/switchboard.train \
    --output_dir data_augmented/babylm_2.5M \
    --ratio 3.0 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_2.5M/babylm_2.5M.vectors.txt \
    --vocab_file models/glove/babylm_2.5M/babylm_2.5M.vocab.txt \
    --top_k 30 \
    --temperature 1.1 \
    --do_eval \
    --verbosity 1 > "logs/babylm_2.5M/switchboard.log" 2>&1 &

wait
echo "Finished 2.5M dataset."