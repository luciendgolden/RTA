# HuLDA

# Query

```sh
poetry run python augment_service/main.py --query "How are you?" --corpus /Users/alexandertampier/Github/HuLDA/data/babylm/clean/10M/bnc_spoken.train

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

# BabyLM 10M (Strict-Small)

| Source | Weight | Domain | Citation | Website | License |
| --- | --- | --- | --- | --- | --- |
| BNC | 8% | Dialogue | BNC Consortium (2007) | [link](http://www.natcorp.ox.ac.uk/) | [link](http://www.natcorp.ox.ac.uk/docs/licence.html) <sup>1</sup> |
| CHILDES | 29% | Dialogue, Child-Directed | MacWhinney (2000) | | [link](https://talkbank.org/share/rules.html) |
| Project Gutenberg | 26% | Fiction, Nonfiction | Gerlach & Font-Clos (2020) | [link](https://github.com/pgcorpus/gutenberg) | [link](https://www.gutenberg.org/policy/license.html) |
| OpenSubtitles | 20% | Dialogue, Scripted | Lison & Tiedermann (2016) | [link](https://opus.nlpl.eu/OpenSubtitles-v2018.php) | Open source |
| Simple English Wikipedia | 15% | Nonfiction | -- | [link](https://dumps.wikimedia.org/simplewiki/20221201/) | [link](https://dumps.wikimedia.org/legal.html) |
| Switchboard | 1% | Dialogue | Godfrey et al. (1992), Stolcke et al., (2000) | [link](http://compprag.christopherpotts.net/swda.html) | [link](http://compprag.christopherpotts.net/swda.html) |

# Augmentation

## BNC

### 50% Augmentation

```sh
poetry run python augment_service/main.py --corpus /Users/alexandertampier/Github/HuLDA/data/babylm/clean/10M/bnc_spoken.train --aug_percentage=0.5
```