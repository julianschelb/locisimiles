# Latin BERT and its tokenizer

Latin BERT ([Bamman and Burns, 2020](https://arxiv.org/abs/2009.10053)) was
trained with a `tensor2tensor` `SubwordTextEncoder`. The HuggingFace
conversions of the model ship the weights and that encoder's vocabulary as
`vocab.txt`, but no tokenizer configuration, so `AutoTokenizer` builds a
WordPiece tokenizer instead.

The two vocabularies have different shapes. Subtokens in a `tensor2tensor`
vocabulary mark word endings with a trailing underscore (`et_`, `que_`,
`faucibus_`) and have no `##` continuation prefix, which WordPiece cannot
segment. Most Latin words then become `[UNK]`:

```text
obstipui steteruntque comae et uox faucibus haesit

WordPiece  →  [UNK] [UNK] [UNK] et [UNK] [UNK] haesit      # 5 of 7 words
original   →  obsti·pui·_  steter·unt·que_  coma·e_  et_
              uo·x_  faucibus_  haesit·_                    # 14 subwords
```

## Using Latin BERT

Download `latin.subword.encoder` from the
[Latin BERT repository](https://github.com/dbamman/latin-bert)
(`models/subword_tokenizer_latin/latin.subword.encoder`) and pass it to the
generator:

```python
from locisimiles.pipeline.generator.contextual_bert import (
    LatinBertContextualCandidateGenerator,
)

generator = LatinBertContextualCandidateGenerator(
    model_name="ashleygong03/bamman-burns-latin-bert",
    subword_encoder_path="models/latin.subword.encoder",
    min_token_length=1,         # no length filter, as in the paper
    use_stopword_filter=False,  # no stopword filter, as in the paper
)
```

The encoder is implemented in pure Python
(`locisimiles.tokenization.latin_bert`), so `tensor2tensor` and TensorFlow are
not required. It reproduces the reference segmentation and uses the same id
layout as `gen_berts.py`: `[PAD] [UNK] [CLS] [SEP] [MASK]` occupy ids 0–4 and
subtoken ids are shifted by `+5`, giving the vocabulary of 32,900 declared in
the checkpoint's `config.json`.

## Tokenizer check

Without `subword_encoder_path`, the generator checks the tokenizer against a
Latin probe sentence and raises if it cannot segment the language:

```python
LatinBertContextualCandidateGenerator(model_name="ashleygong03/bamman-burns-latin-bert")
# ValueError: The tokenizer loaded for 'ashleygong03/bamman-burns-latin-bert' maps
# 71% of Latin words to [UNK]. ... Pass subword_encoder_path=<path to
# latin.subword.encoder> to use the model's original encoder ...
```

Pass `check_tokenizer=False` to skip the check.

## Vocabulary coverage

Embedding checkpoints expect text in the form they were trained on. The Latin
word vectors, for example, are trained on u/i-normalized lemmas (`uolo`, `uel`,
`ciuilis`), while CLTK's lemmatizer produces v/j spellings and homonym indices
(`volo`, `vel`, `venus2`). `Word2VecCandidateGenerator` lemmatizes and then
normalizes orthography, in that order, before looking a token up.

When a checkpoint covers few of the tokens it is given, the generator reports
it:

```text
UserWarning: Word2Vec generator (latin_w2v_bamman_lemma300_100_1.model): only
37.5% of tokens are in the vocabulary ... frequent OOV: volo (3165), vel (1993)
```

`locisimiles.diagnostics.vocab_coverage` computes the same figures directly for
any token stream and vocabulary.
