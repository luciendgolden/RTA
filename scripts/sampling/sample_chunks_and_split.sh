# word count and line count for babylm original 10M data
echo "Counting words and lines for babylm original 10M data..."
wc -wl data/babylm/original/10M/*

# Make data for 9M
echo "Sampling 9M data..."

## bnc
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/bnc_spoken.train --output_dir data_sampled/babylm_9M --p_keep 0.9 --p_keep_dev 0.1 --split_at 1000 --seed 4

## childes
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/childes.train --output_dir data_sampled/babylm_9M --p_keep 0.9 --p_keep_dev 0.1 --split_at 1000 --seed 2

## gutenberg
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/gutenberg.train --output_dir data_sampled/babylm_9M --p_keep 0.9 --p_keep_dev 0.1 --split_at 500 --seed 1

## openSubtitles
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/open_subtitles.train --output_dir data_sampled/babylm_9M --p_keep 0.9 --p_keep_dev 0.1 --split_at 500 --seed 8

## simple english wiki
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/simple_wiki.train --output_dir data_sampled/babylm_9M --p_keep 0.9 --p_keep_dev 0.1 --split_at 500 --seed 10

## switchboard
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/switchboard.train --output_dir data_sampled/babylm_9M --p_keep 0.9 --p_keep_dev 0.1 --split_at 200 --seed 9

rm data_sampled/babylm_9M/*.dev
rm data_sampled/babylm_9M/*.test
wc -wl data_sampled/babylm_9M/*.train

# Make data for 7.5M
echo "Sampling 7.5M data..."

## bnc
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/bnc_spoken.train --output_dir data_sampled/babylm_7.5M --p_keep 0.75 --p_keep_dev 0.1 --split_at 1000 --seed 3

## childes
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/childes.train --output_dir data_sampled/babylm_7.5M --p_keep 0.75 --p_keep_dev 0.1 --split_at 1000 --seed 4

## gutenberg
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/gutenberg.train --output_dir data_sampled/babylm_7.5M --p_keep 0.75 --p_keep_dev 0.1 --split_at 500 --seed 8

## openSubtitles
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/open_subtitles.train --output_dir data_sampled/babylm_7.5M --p_keep 0.75 --p_keep_dev 0.1 --split_at 500 --seed 0

## simple english wiki
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/simple_wiki.train --output_dir data_sampled/babylm_7.5M --p_keep 0.75 --p_keep_dev 0.1 --split_at 500 --seed 1

## switchboard
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/switchboard.train --output_dir data_sampled/babylm_7.5M --p_keep 0.75 --p_keep_dev 0.1 --split_at 200 --seed 2

rm data_sampled/babylm_7.5M/*.dev
rm data_sampled/babylm_7.5M/*.test
wc -wl data_sampled/babylm_7.5M/*.train

# Make data for 5M
echo "Sampling 5M data..."

## bnc
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/bnc_spoken.train --output_dir data_sampled/babylm_5M --p_keep 0.5 --p_keep_dev 0.1 --split_at 1000 --seed 3

## childes
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/childes.train --output_dir data_sampled/babylm_5M --p_keep 0.5 --p_keep_dev 0.1 --split_at 1000 --seed 5

## childrens gutenberg
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/gutenberg.train --output_dir data_sampled/babylm_5M --p_keep 0.5 --p_keep_dev 0.1 --split_at 500 --seed 5

## openSubtitles
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/open_subtitles.train --output_dir data_sampled/babylm_5M --p_keep 0.5 --p_keep_dev 0.1 --split_at 500 --seed 3

## simple english wiki
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/simple_wiki.train --output_dir data_sampled/babylm_5M --p_keep 0.5 --p_keep_dev 0.1 --split_at 500 --seed 6

## switchboard
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/switchboard.train --output_dir data_sampled/babylm_5M --p_keep 0.5 --p_keep_dev 0.1 --split_at 200 --seed 6

rm data_sampled/babylm_5M/*.dev
rm data_sampled/babylm_5M/*.test
wc -wl data_sampled/babylm_5M/*.train

# Make data for 2.5M
echo "Sampling 2.5M data..."

## bnc
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/bnc_spoken.train --output_dir data_sampled/babylm_2.5M --p_keep 0.25 --p_keep_dev 0.1 --split_at 1000 --seed 2

## childes
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/childes.train --output_dir data_sampled/babylm_2.5M --p_keep 0.25 --p_keep_dev 0.1 --split_at 1000 --seed 14

## childrens gutenberg
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/gutenberg.train --output_dir data_sampled/babylm_2.5M --p_keep 0.25 --p_keep_dev 0.1 --split_at 500 --seed 2

## openSubtitles
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/open_subtitles.train --output_dir data_sampled/babylm_2.5M --p_keep 0.25 --p_keep_dev 0.1 --split_at 500 --seed 0

## simple english wiki
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/simple_wiki.train --output_dir data_sampled/babylm_2.5M --p_keep 0.25 --p_keep_dev 0.1 --split_at 500 --seed 2

## switchboard
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/switchboard.train --output_dir data_sampled/babylm_2.5M --p_keep 0.25 --p_keep_dev 0.1 --split_at 200 --seed 1

rm data_sampled/babylm_2.5M/*.dev
rm data_sampled/babylm_2.5M/*.test
wc -wl data_sampled/babylm_2.5M/*.train

# Make data for 1M
echo "Sampling 1M data..."

## bnc
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/bnc_spoken.train --output_dir data_sampled/babylm_1M --p_keep 0.1 --p_keep_dev 0.1 --split_at 1000 --seed 4

## childes
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/childes.train --output_dir data_sampled/babylm_1M --p_keep 0.1 --p_keep_dev 0.1 --split_at 1000 --seed 11

## childrens gutenberg
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/gutenberg.train --output_dir data_sampled/babylm_1M --p_keep 0.1 --p_keep_dev 0.1 --split_at 500 --seed 9

## openSubtitles
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/open_subtitles.train --output_dir data_sampled/babylm_1M --p_keep 0.1 --p_keep_dev 0.1 --split_at 500 --seed 4

## simple english wiki
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/simple_wiki.train --output_dir data_sampled/babylm_1M --p_keep 0.1 --p_keep_dev 0.1 --split_at 500 --seed 8

## switchboard
python3 augment_service/sampling/sample_chunks_and_split.py --input_file data/babylm/original/10M/switchboard.train --output_dir data_sampled/babylm_1M --p_keep 0.1 --p_keep_dev 0.1 --split_at 200 --seed 1

rm data_sampled/babylm_1M/*.dev
rm data_sampled/babylm_1M/*.test
wc -wl data_sampled/babylm_1M/*.train