# RecombiText Augmentation (RTA)
We introduce RecombiText Augmentation (RTA), a novel purely statistical NLP method for compositional data augmentation for data-efficient LLM pre-training in low-resource scenarios. RTA identifies lexically and semantically similar sentences within the corpus and generates synthetic sentence pairs from them while preserving underlying patterns from the corpus. We pre-train GPT-2 and RoBERTa language models on a domain-specific, low-resource corpus of 10 million words, with different proportions of augmented data. We compare our RTA-augmented model variants to a baseline model trained on the full original dataset. Zero-shot results show that the language models pre-trained on synthetic data improve in entity tracking, self-paced reading, and morphological generalization benchmarks. In other tasks, the performance is comparable to the baseline model. We demonstrate that it is possible to expand low-resource datasets by two- to four-fold without compromising benchmark performance, solely through statistical processing of the available data.

# Query

```sh
poetry run python augment_service/main.py \
        --query "Two months passed, and spring deepened into summer." \
        --input_file data_cleaned/babylm_10M/gutenberg.train \
        --context_window_size 3 \
        --context_window_threshold 0.6 \
        --rank_constant 60 \
        --embeddings_file models/glove/babylm_10M/babylm_10M.vectors.txt \
        --vocab_file models/glove/babylm_10M/babylm_10M.vocab.txt \
        --top_k 10 \
        --temperature 0.7 \
        --verbosity 2

2025-08-26 18:45:19,168 - INFO - Starting Searching and augmenting 'Two months passed, and spring deepened into summer.'...
2025-08-26 18:45:19,219 - INFO - --------------------
2025-08-26 18:45:19,229 - INFO - Top 10/99 RRF results: 'Two months passed, and spring deepened into summer.'
2025-08-26 18:45:19,229 - INFO - 1. Two months passed, and spring deepened into summer.  (_rrf_score: 0.095)
2025-08-26 18:45:19,229 - INFO - 2. The spring months had passed, the apple-trees had blossomed, and the young apples had formed.  (_rrf_score: 0.045)
2025-08-26 18:45:19,229 - INFO - 3. They also pointed out that as _The Revenge_ had six feet of water in the hold and three shots under water, but weakly stopped, she must needs sink in the first heavy sea; which indeed happened a few days later.  (_rrf_score: 0.045)
2025-08-26 18:45:19,229 - INFO - 4. “Two or three months.” (_rrf_score: 0.043)
2025-08-26 18:45:19,229 - INFO - 5. When night fell we retired a mile or two from the river and went into camp.  (_rrf_score: 0.043)
2025-08-26 18:45:19,229 - INFO - 6. Another two months passed.  (_rrf_score: 0.042)
2025-08-26 18:45:19,229 - INFO - 7. She was walking abroad this morning between four and five o'clock at some distance from here." (_rrf_score: 0.042)
2025-08-26 18:45:19,230 - INFO - 8. Weeks passed by and lengthened into months with no word either of Sobrinini or of the ex-slaves.  (_rrf_score: 0.040)
2025-08-26 18:45:19,230 - INFO - 9. Half an hour later he saw a boat row off to one of them, which he had already singled out, from its size and general appearance, as being that of Hassan; ten minutes later he saw it returning.  (_rrf_score: 0.040)
2025-08-26 18:45:19,230 - INFO - 10. summer away cake winter swans spring flew bread leaves
 (_rrf_score: 0.038)
2025-08-26 18:45:19,231 - DEBUG - Selected candidate (top-10 sampling): _rrf_score=0.042, prob=0.099 «She was walking abroad this morning between four and five o'clock at some distance from here."»
2025-08-26 18:45:19,234 - DEBUG - Window size 3: Best score = 0.7023, Window = w1['two', 'months', 'passed'] w2['five', "o'clock", 'at'], Pivot = (1, 10) s1['months'] s2['o'clock']
2025-08-26 18:45:19,234 - INFO - --------------------
2025-08-26 18:45:19,234 - INFO - Retry 1/99: Crossover successful
2025-08-26 18:45:19,234 - INFO - Reference Sentence: Two months passed, and spring deepened into summer.
2025-08-26 18:45:19,234 - INFO - Candidate Sentence: She was walking abroad this morning between four and five o'clock at some distance from here."
2025-08-26 18:45:19,234 - INFO - Aug1: Two o'clock at some distance from here."
2025-08-26 18:45:19,234 - INFO - Aug2: She was walking abroad this morning between four and five months passed, and spring deepened into summer.
2025-08-26 18:45:19,234 - INFO - Finished Searching and augmenting 'Two months passed, and spring deepened into summer.'
2025-08-26 18:45:19,234 - INFO - Elapsed time: 0.07 seconds
```

# Data

| Source | Weight | Domain | Citation | Website | License |
| --- | --- | --- | --- | --- | --- |
| BNC | 8% | Dialogue | BNC Consortium (2007) | [link](http://www.natcorp.ox.ac.uk/) | [link](http://www.natcorp.ox.ac.uk/docs/licence.html) <sup>1</sup> |
| CHILDES | 29% | Dialogue, Child-Directed | MacWhinney (2000) | | [link](https://talkbank.org/share/rules.html) |
| Project Gutenberg | 26% | Fiction, Nonfiction | Gerlach & Font-Clos (2020) | [link](https://github.com/pgcorpus/gutenberg) | [link](https://www.gutenberg.org/policy/license.html) |
| OpenSubtitles | 20% | Dialogue, Scripted | Lison & Tiedermann (2016) | [link](https://opus.nlpl.eu/OpenSubtitles-v2018.php) | Open source |
| Simple English Wikipedia | 15% | Nonfiction | -- | [link](https://dumps.wikimedia.org/simplewiki/20221201/) | [link](https://dumps.wikimedia.org/legal.html) |
| Switchboard | 1% | Dialogue | Godfrey et al. (1992), Stolcke et al., (2000) | [link](http://compprag.christopherpotts.net/swda.html) | [link](http://compprag.christopherpotts.net/swda.html) |

# Data Augmentation

```sh
poetry run python augment_service/main.py \
    --input_file data_cleaned/babylm_10M/gutenberg.train \
    --output_dir data_augmented \
    --ratio 1.00 \
    --context_window_size 3 \
    --context_window_threshold 0.6 \
    --rank_constant 60 \
    --embeddings_file models/glove/babylm_10M/babylm_10M.vectors.txt \
    --vocab_file models/glove/babylm_10M/babylm_10M.vocab.txt \
    --top_k 10 \
    --temperature 0.7 \
    --do_eval \
```
