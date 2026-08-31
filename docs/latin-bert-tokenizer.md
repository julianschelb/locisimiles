# Latin BERT and its tokenizer

Latin BERT ([Bamman and Burns, 2020](https://arxiv.org/abs/2009.10053)) was
trained with a `tensor2tensor` `SubwordTextEncoder`. The public HuggingFace
conversions ship the model weights and that encoder's vocabulary as
`vocab.txt`, but **no tokenizer configuration**.

This combination fails quietly. `AutoTokenizer.from_pretrained` sees a
`vocab.txt` and builds a **WordPiece** tokenizer, but the vocabulary is not a
WordPiece vocabulary: its subtokens mark word endings with a trailing
underscore (`et_`, `que_`, `faucibus_`) and carry no `##` continuation prefix.
WordPiece cannot segment against it, so it falls back to `[UNK]` — for about
**60–70 % of Latin words**. No error is raised, and the model still produces
scores that look plausible.

```text
obstipui steteruntque comae et uox faucibus haesit

AutoTokenizer  →  [UNK] [UNK] [UNK] et [UNK] [UNK] haesit      # 5 of 7 words lost
correct        →  obsti·pui·_  steter·unt·que_  coma·e_  et_
                  uo·x_  faucibus_  haesit·_                    # 14 subwords, 0 [UNK]
```

## What this package does

Constructing a contextual generator validates the tokenizer against a Latin
probe sentence and **refuses to run** if it cannot segment the language:

```python
from locisimiles.pipeline.generator.contextual_bert import (
    LatinBertContextualCandidateGenerator,
)

LatinBertContextualCandidateGenerator(model_name="ashleygong03/bamman-burns-latin-bert")
# ValueError: The tokenizer loaded for 'ashleygong03/bamman-burns-latin-bert' maps
# 71% of Latin words to [UNK]. ... Pass subword_encoder_path=<path to
# latin.subword.encoder> to use the model's original encoder ...
```

## Using Latin BERT correctly

Download `latin.subword.encoder` from the
[Latin BERT repository](https://github.com/dbamman/latin-bert) (
`models/subword_tokenizer_latin/latin.subword.encoder`) and point the generator
at it:

```python
generator = LatinBertContextualCandidateGenerator(
    model_name="ashleygong03/bamman-burns-latin-bert",
    subword_encoder_path="models/latin.subword.encoder",
    min_token_length=1,        # paper-faithful: no length filter
    use_stopword_filter=False, # paper-faithful: no stopword filter
)
```

The encoder is reimplemented in pure Python
(`locisimiles.tokenization.latin_bert`), so no TensorFlow or `tensor2tensor`
dependency is required. It reproduces the reference segmentation exactly, and
applies the same id layout as `gen_berts.py`: `[PAD] [UNK] [CLS] [SEP] [MASK]`
occupy ids 0–4 and every subtoken id is shifted by `+5`, giving a vocabulary of
32,900 that matches the checkpoint's `config.json`.

## Why it matters

Evaluated on the *Loci Similes* benchmark, the tokenizer alone accounts for the
difference between a baseline that looks unusable and one that is competitive:

| | broken tokenizer | original encoder |
|---|---|---|
| Retrieval, pooled Recall@100 | 0.147 | **0.616** |
| Cross-encoder, verbatim F1 | 9.53 | **77.82** |

## A related trap: word vectors and orthography

The same class of mismatch affects the static Latin word vectors. The published
checkpoints are trained on **u/i-normalized lemmas** (`uolo`, `uel`, `ciuilis`),
whereas CLTK's lemmatizer emits v/j spellings and homonym indices (`volo`,
`vel`, `venus2`). Looking the latter up in the former simply misses.

`Word2VecCandidateGenerator` therefore lemmatizes with CLTK and *then*
normalizes orthography — that order matters — and warns when few tokens are
found in the vocabulary:

```text
UserWarning: Word2Vec generator (latin_w2v_bamman_lemma300_100_1.model): only
37.5% of tokens are in the vocabulary ... frequent OOV: volo (3165), vel (1993)
```

That warning is the generic form of the check: any embedding checkpoint whose
preprocessing disagrees with your text will show up as low coverage rather than
as an error.
