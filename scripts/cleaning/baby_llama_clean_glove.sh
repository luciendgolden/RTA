# Tokenize for 1M
echo "Tokenizing 1M data..."
poetry run python augment_service/clean/baby_llama_clean_glove.py --input_dir data_sampled/babylm_1M --output_dir data_glove/babylm_1M

# Tokenize for 2.5M
echo "Tokenizing 2.5M data..."
poetry run python augment_service/clean/baby_llama_clean_glove.py --input_dir data_sampled/babylm_2.5M --output_dir data_glove/babylm_2.5M

# Tokenize for 5M
echo "Tokenizing 5M data..."
poetry run python augment_service/clean/baby_llama_clean_glove.py --input_dir data_sampled/babylm_5M --output_dir data_glove/babylm_5M

# Tokenize for 7.5M
echo "Tokenizing 7.5M data..."
poetry run python augment_service/clean/baby_llama_clean_glove.py --input_dir data_sampled/babylm_7.5M --output_dir data_glove/babylm_7.5M

# Tokenize for 9M
echo "Tokenizing 9M data..."
poetry run python augment_service/clean/baby_llama_clean_glove.py --input_dir data_sampled/babylm_9M --output_dir data_glove/babylm_9M

# Tokenize for 10M
echo "Tokenizing 10M data..."
poetry run python augment_service/clean/baby_llama_clean_glove.py --input_dir data/babylm/original/10M --output_dir data_glove/babylm_10M