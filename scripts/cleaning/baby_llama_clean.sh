# Clean data for 1M
echo "Cleaning data_sampled/babylm_1M..."
wc -wl data_sampled/babylm_1M/*
python3 augment_service/clean/baby_llama_clean.py --input_dir data_sampled/babylm_1M --output_dir data_cleaned/babylm_1M
wc -wl data_cleaned/babylm_1M/*.train

# Clean data for 2.5M
echo "Cleaning data_sampled/babylm_2.5M..."
wc -wl data_sampled/babylm_2.5M/*
python3 augment_service/clean/baby_llama_clean.py --input_dir data_sampled/babylm_2.5M --output_dir data_cleaned/babylm_2.5M
wc -wl data_cleaned/babylm_2.5M/*.train

# Clean data for 5M
echo "Cleaning data_sampled/babylm_5M..."
wc -wl data_sampled/babylm_5M/*
python3 augment_service/clean/baby_llama_clean.py --input_dir data_sampled/babylm_5M --output_dir data_cleaned/babylm_5M
wc -wl data_cleaned/babylm_5M/*.train

# Clean data for 7.5M
echo "Cleaning data_sampled/babylm_7.5M..."
wc -wl data_sampled/babylm_7.5M/*
python3 augment_service/clean/baby_llama_clean.py --input_dir data_sampled/babylm_7.5M --output_dir data_cleaned/babylm_7.5M
wc -wl data_cleaned/babylm_7.5M/*.train

# Clean data for 9M
echo "Cleaning data_sampled/babylm_9M..."
wc -wl data_sampled/babylm_9M/*
python3 augment_service/clean/baby_llama_clean.py --input_dir data_sampled/babylm_9M --output_dir data_cleaned/babylm_9M
wc -wl data_cleaned/babylm_9M/*.train

# Clean data for 10M
echo "Cleaning data/babylm/original/10M..."
wc -wl data/babylm/original/10M/*
python3 augment_service/clean/baby_llama_clean.py --input_dir data/babylm/original/10M --output_dir data_cleaned/babylm_10M
wc -wl data_cleaned/babylm_10M/*.train
