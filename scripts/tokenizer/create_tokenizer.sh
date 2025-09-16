# tokenizer for babylm base 1M
poetry run python pretrain_service/tokenizer_creation/create_tokenizer.py \
    --input_path data_augmented/babylm_1M/full_train.txt \
    --validation_path data_augmented/babylm_1M/full_train.txt \
    --vocab_path pretrain_service/tokenizers/babylm_1M/tokenizer_babylm1M.json \
    --output_dir pretrain_service/tokenizers/babylm_1M \
    --vocab_size 16384 \
    --min_frequency 10

# tokenizer for babylm base 2.5M
poetry run python pretrain_service/tokenizer_creation/create_tokenizer.py \
    --input_path data_augmented/babylm_2.5M/full_train.txt \
    --validation_path data_augmented/babylm_2.5M/full_train.txt \
    --vocab_path pretrain_service/tokenizers/babylm_2.5M/tokenizer_babylm2.5M.json \
    --output_dir pretrain_service/tokenizers/babylm_2.5M \
    --vocab_size 16384 \
    --min_frequency 10

# tokenizer for babylm base 5M
poetry run python pretrain_service/tokenizer_creation/create_tokenizer.py \
    --input_path data_augmented/babylm_5M/full_train.txt \
    --validation_path data_augmented/babylm_5M/full_train.txt \
    --vocab_path pretrain_service/tokenizers/babylm_5M/tokenizer_babylm5M.json \
    --output_dir pretrain_service/tokenizers/babylm_5M \
    --vocab_size 16384 \
    --min_frequency 10

# tokenizer for babylm base 7.5M
poetry run python pretrain_service/tokenizer_creation/create_tokenizer.py \
    --input_path data_augmented/babylm_7.5M/full_train.txt \
    --validation_path data_augmented/babylm_7.5M/full_train.txt \
    --vocab_path pretrain_service/tokenizers/babylm_7.5M/tokenizer_babylm7.5M.json \
    --output_dir pretrain_service/tokenizers/babylm_7.5M \
    --vocab_size 16384 \
    --min_frequency 10

# tokenizer for babylm base 9M
poetry run python pretrain_service/tokenizer_creation/create_tokenizer.py \
    --input_path data_augmented/babylm_9M/full_train.txt \
    --validation_path data_augmented/babylm_9M/full_train.txt \
    --vocab_path pretrain_service/tokenizers/babylm_9M/tokenizer_babylm9M.json \
    --output_dir pretrain_service/tokenizers/babylm_9M \
    --vocab_size 16384 \
    --min_frequency 10

# tokenizer for babylm base 10M
poetry run python pretrain_service/tokenizer_creation/create_tokenizer.py \
    --input_path data_cleaned/babylm_10M/full_train.txt \
    --validation_path data_cleaned/babylm_10M/full_train.txt \
    --vocab_path pretrain_service/tokenizers/babylm_10M/tokenizer_babylm10M.json \
    --output_dir pretrain_service/tokenizers/babylm_10M \
    --vocab_size 16384 \
    --min_frequency 10

# tokenizer for rta10m
poetry run python pretrain_service/tokenizer_creation/create_tokenizer.py \
    --input_path data_augmented/rta_10M/full_train.txt \
    --validation_path data_augmented/rta_10M/full_train.txt \
    --vocab_path pretrain_service/tokenizers/rta_10M/tokenizer_rta10M.json \
    --output_dir pretrain_service/tokenizers/rta_10M \
    --vocab_size 16384 \
    --min_frequency 10
