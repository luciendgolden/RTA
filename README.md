# RecombiText Augmentation (RTA)

Based on the Strict-Small BabyLM 3rd iteration, we introduce RecombiText Augmentation (RTA), a novel non-neural NLP method for compositional data augmentation that sets steps towards efficient LLM pre-training in low-resource scenarios. RTA identifies lexically and semantically similar sentences within the corpus and applies a one-point crossover to generate synthetic sentence pairs while preserving underlying patterns from the corpus.  We pre-train an encoder-based language model, RoBERTa-base, on a 10-million-word corpus. A baseline is created and compared with the RTA-augmented variants. Dataset quality is assessed via perplexity (PPL). Language model efficiency is evaluated on zero-shot and fine-tuning tasks. Zero-shot results show that RoBERTa improves significantly in Entity Tracking and morphological generalization (WUGs) but shows a slight decline in grammar tasks (BLiMP, BLiMP Supplement). Fine-tuning results yield improvements in reading comprehension (MultiRC), recognizing text entailments (RTE), and coreference resolution (WSC). Comparable performances to the baseline are achieved in the other tasks. We demonstrate that RTA is capable of effectively enhancing LLM pre-training datasets in low-resource scenarios and offers a low-dependency alternative to existing methods.

# Query

```sh
poetry run python augment_service/main.py --query "How are you?" --corpus /data/babylm/clean/10M/bnc_spoken.train

2025-06-02 14:05:10,078 - DEBUG - --------------------
2025-06-02 14:05:10,078 - INFO - Top 10 search results for query: 'How are you?'
2025-06-02 14:05:10,078 - INFO - 1. How are you? (Cosine: 0.967)
2025-06-02 14:05:10,078 - INFO - 2. How are you doing for those? (Cosine: 0.895)
2025-06-02 14:05:10,078 - INFO - 3. How are we doing? (Cosine: 0.859)
2025-06-02 14:05:10,078 - INFO - 4. How much are they? (Cosine: 0.836)
2025-06-02 14:05:10,078 - INFO - 5. How old are you now Belinda? (Cosine: 0.836)
2025-06-02 14:05:10,078 - INFO - 6. How are you, alright? (Cosine: 0.835)
2025-06-02 14:05:10,078 - INFO - 7. How many of you know that? (Cosine: 0.829)
2025-06-02 14:05:10,078 - INFO - 8. How are you working it out? (Cosine: 0.818)
2025-06-02 14:05:10,078 - INFO - 9. How do you
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

# Augmentation

```sh
poetry run python augment_service/main.py --corpus /data/babylm/clean/10M/bnc_spoken.train --aug_percentage=0.5
```