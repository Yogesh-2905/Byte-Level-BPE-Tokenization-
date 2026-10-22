
# Byte-Level-BPE-Tokenization-

Built a custom Byte-Level Byte Pair Encoding (BPE) tokenizer from scratch using Python. Explored byte-level tokenization, frequency-based merging, vocabulary construction, token IDs, and text encoding and decoding to understand how tokenizers prepare text for Large Language Models (LLMs).

## Project Overview

Tokenization is one of the first steps in processing text for a Large Language Model (LLM). Before a model can process text, the input must be converted into tokens and then represented as numerical token IDs.

In this project, I explored how tokenization works by implementing a basic whitespace tokenizer, experimenting with a pretrained tokenizer from Hugging Face, and building a custom Byte-Level Byte Pair Encoding (BPE) tokenizer from scratch.

The project helped me understand how text is transformed into numerical representations before being processed by an LLM, and why tokenization matters for model performance, context windows, and API costs.

## Objectives

- Understand the fundamentals of text tokenization.
- Learn the difference between characters, words, tokens, and token IDs.
- Implement a basic whitespace tokenizer using Python.
- Experiment with a pretrained BERT tokenizer using Hugging Face Transformers.
- Build a custom Byte-Level BPE tokenizer without using pretrained tokenizer rules.
- Compare tokenization behavior across different approaches.
- Understand the relationship between tokenization, LLMs, and Retrieval-Augmented Generation (RAG).

## Technologies Used

- Python
- Jupyter Notebook
- Hugging Face Transformers
- BERT tokenizer (bert-base-uncased)
- Regular expressions
- Python collections.Counter
- JSON for saving and loading tokenizer data

## Project Structure

```text
tokenizer-for-llms/
│
├── README.md
├── tokenizer_experiments.ipynb
├── custom_byte_level_bpe.json
└── requirements.txt
```

*Note: Adjust the filenames above to match the files actually included in your GitHub repository.*

# Part 1: Understanding Tokenization

### 1. What is tokenization?

Tokenization is the process of dividing text into smaller units called tokens and mapping those tokens to numerical IDs.

For example, a tokenizer might process:
```text
Input: I love AI!
Tokens: ["I", "love", "AI", "!"]
```

The actual tokens depend on the tokenizer being used.

LLMs operate on token IDs rather than directly processing raw text. These IDs are subsequently mapped to learned vector representations called embeddings.

### 2. Character vs. Word vs. Token vs. Token ID

- Character: A text symbol, such as A, b, or !.
- Word: A linguistic unit, such as learning.
- Token: A unit selected by a tokenizer. It may represent a word, part of a word, punctuation, whitespace, or bytes.
- Token ID: An integer assigned to a token in a tokenizer's vocabulary.
A word does not necessarily correspond to one token. For example, a subword tokenizer may split an unfamiliar word into several smaller pieces.

### 3. Types of tokenization

#### Whitespace tokenization

Splits text at whitespace characters.

Example:
```text
Input: I love machine learning!
Tokens: ["I", "love", "machine", "learning!"]
```

This is simple to implement, but punctuation remains attached to words, and different word forms become separate vocabulary entries.

#### Subword tokenization

Splits text into units smaller than or equal to a word. Frequently occurring words may remain whole, while unfamiliar words can be represented using smaller pieces.

#### Byte Pair Encoding (BPE)

BPE starts with smaller units and repeatedly merges frequently occurring adjacent pairs. The learned merge rules are then used to tokenize new text.

In a byte-level implementation, the initial vocabulary contains all 256 possible byte values. This allows text from different languages and emoji to be represented through UTF-8 bytes.

#### WordPiece

WordPiece is a subword tokenization method used by BERT. It uses a learned vocabulary and segmentation procedure to represent text as vocabulary tokens. It is related to BPE but does not use the same merge-learning procedure.

### 4. Why can one word be split into multiple tokens?

A tokenizer may not contain a complete word in its vocabulary. Instead of treating every unfamiliar word as completely unknown, a subword tokenizer may divide it into smaller known units.

For example, a hypothetical tokenizer might split:
```text
unhappiness → un + happi + ness
```

This is an illustrative example; the actual segmentation depends on the tokenizer.

Subword tokenization helps balance vocabulary size and the ability to represent unfamiliar words.

### 5. How tokenization affects context windows and API costs

LLMs generally have a context-window limit measured in tokens, not words or characters.

If a prompt contains many tokens, fewer tokens remain available for the response and other context. For APIs that charge based on token usage, input and output token counts can affect the total cost.

Different tokenizers may represent the same text using different numbers of tokens, so tokenization can influence context usage, latency, and cost.

# Part 2: My Custom Byte-Level BPE Tokenizer

### 1. What I built

I implemented a Byte-Level Byte Pair Encoding (BPE) tokenizer in Python without loading pretrained tokenizer vocabulary or merge rules. The implementation is contained in my Jupyter Notebook.

The tokenizer follows this workflow:
1. Normalize case: Convert the input text to lowercase before tokenization.
2. Pre-tokenize: Use a regular expression to separate word-like pieces, punctuation groups, and whitespace groups.
3. Convert to UTF-8 bytes: Convert each pre-tokenized piece into its underlying bytes.
4. Initialize the vocabulary: Start with all 256 possible byte values, assigning one token ID to each byte.
5. Count adjacent pairs: Count how often neighboring token IDs occur together in the training sequences.
6. Learn merges: Repeatedly merge the most frequent adjacent pair and add the resulting byte sequence to the vocabulary.
7. Encode new text: Apply the learned merge rules in the order in which they were learned.
8. Save and reload: Save the vocabulary and merge rules to a JSON file and check that reloading preserves tokenization.

### 2. Results recorded in my notebook

The following values are from the executed notebook, not estimates.
| Measurement | Recorded result |
|---|---|
| Target vocabulary size | 3,000 |
| Number of pre-tokenized training pieces | 41,365 |
| Initial byte vocabulary | 256 |
| Final vocabulary size | 2,448 |
| Learned merge rules | 2,192 |
| Training reconstruction test | Passed |
| Case-insensitive tokenization tests | Passed |
| Save/reload token-ID verification | Passed |

Why did the vocabulary stop at 2,448 instead of 3,000?

The training loop stopped because no adjacent pairs remained to merge in the training sequences. The target vocabulary size is a maximum stopping goal, not a guarantee that the vocabulary will reach that exact size. This result is specific to the training corpus and this implementation.

### 3. Examples from my tokenizer

The notebook recorded these results:
| Input | Token IDs | Token count |
|---|---|---:|
| Token | [521] | 1 |
| TOKEN | [521] | 1 |
| Tokenization | [521, 593, 995] | 3 |
| TOKENIZATION | [521, 593, 995] | 3 |
| Tokenizer | [521, 593, 267] | 3 |
| tokens | [1945] | 1 |
| Hello World! | [2021, 32, 2275, 33] | 4 |

These results show that the same spelling with different capitalization receives the same token IDs because the tokenizer lowercases text before encoding. BPE can also learn a whole word as one token when its merge rules support that result, as happened with Token and tokens in this training run.

### 4. Important limitation: capitalization and exact reconstruction

The normal encode() method lowercases the input. Therefore, decode(tokenizer.encode("Hello")) returns normalized text ("hello"), not the original capitalization ("Hello").

The notebook also demonstrates a separate in-memory record that retains the original text for a specific encoded sequence. This is a demonstration technique, not information stored in the token IDs. The original-text records are not included in the saved tokenizer JSON, so they are not available after loading the tokenizer into a fresh session.

This distinction matters: case-insensitive token IDs and exact original-text reconstruction are two different requirements. To recover original capitalization from token IDs alone, the tokenizer would need to preserve that information rather than discard it during normalization.

### 5. Multilingual text and byte-level behavior

The notebook tests Tamil text, emoji, punctuation, numbers, and mixed text. UTF-8 byte-level representation gives the tokenizer a way to represent these inputs without requiring a separate initial token for every possible character.

When inspecting an individual token, a byte sequence may be only part of a multi-byte UTF-8 character. Displaying that individual token as text can therefore show a replacement symbol (�). That does not necessarily mean the complete encoded text is corrupted. The complete sequence must be combined into bytes and decoded together. The notebook's full-text reconstruction test is the more meaningful check.

### 6. What these tests do and do not prove

The successful tests demonstrate that the tokenizer learns and reuses merge rules, produces repeatable IDs for case variants, and can save and reload its vocabulary and merge rules without changing the tested token IDs.

They do not prove that this tokenizer is more accurate, faster, or more efficient than a production tokenizer. Those claims require a controlled comparison on the same test texts and a clear measurement method.

# Part 3: Comparing My Tokenizer with Hugging Face AutoTokenizer

### 1. What is AutoTokenizer?

AutoTokenizer is a Hugging Face Transformers utility that loads the tokenizer configuration associated with a pretrained model. In this comparison, AutoTokenizer.from_pretrained("bert-base-uncased") loads the tokenizer used by BERT Base Uncased.

The BERT tokenizer uses WordPiece tokenization and a pretrained vocabulary. My custom tokenizer learns byte-pair merges from my own training corpus. These are different tokenization approaches, trained with different data and rules.

### 2. What should be compared?

For a fair first experiment, send the same text examples to both tokenizers and compare:
- The tokens produced by each tokenizer.
- The number of tokens produced for each text.
- How each handles capitalization.
- How each handles punctuation, numbers, Tamil text, and emoji.
- Whether each tokenizer can represent the test input without an unknown token.
- How token counts vary across the chosen examples.
Token IDs from the two tokenizers must not be compared as if they have the same meaning. Token ID 521 in my tokenizer and token ID 521 in BERT refer to entries in two different vocabularies.

### 3. Run the comparison in Jupyter Notebook

Install the Hugging Face Transformers library in a notebook cell:
```python
%pip install transformers
```

Then run the following in a new cell after training your custom tokenizer. It assumes your custom tokenizer is stored in the variable tokenizer, as in the current notebook.
```python
from transformers import AutoTokenizer
import pandas as pd

# Load the pretrained BERT tokenizer.
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

comparison_texts = [
    "Token",
    "TOKEN",
    "Tokenization and TOKENIZATION",
    "Hello World!",
    "Python_3 is GREAT.",
    "Don't lose punctuation, spaces, or numbers: 123!",
    "தமிழ் மொழி 😊",
    "Hello தமிழ் 😊 123!",
]

comparison_rows = []

for text in comparison_texts:
    # Custom tokenizer: IDs are generated by your own trained vocabulary.
    custom_ids = tokenizer.encode(text)

    # BERT tokenizer: do not add [CLS] and [SEP] for a direct content comparison.
    bert_ids = bert_tokenizer.encode(text, add_special_tokens=False)
    bert_tokens = bert_tokenizer.convert_ids_to_tokens(bert_ids)

    comparison_rows.append({
        "input_text": text,
        "custom_tokens": [
            tokenizer.get_token_bytes(token_id).decode("utf-8", errors="replace")
            for token_id in custom_ids
        ],
        "custom_token_count": len(custom_ids),
        "bert_tokens": bert_tokens,
        "bert_token_count": len(bert_ids),
        "bert_unknown_token_present": (
            bert_tokenizer.unk_token_id in bert_ids
        ),
    })

comparison_df = pd.DataFrame(comparison_rows)
display(comparison_df)
```

Note: Individual custom tokens that contain only part of a multi-byte UTF-8 character may display as � in the table. The underlying bytes are what the custom tokenizer stores; the display form is only a readable approximation.

### 4. How to interpret the comparison

After running the cell, use the actual table output to discuss these questions:
1. Which tokenizer produced fewer tokens for each example?
2. Did the custom tokenizer map capitalization variants to the same IDs?
3. Did BERT split the same words into different subwords?
4. Did either tokenizer produce an unknown token for any example?
5. How did the token counts differ for Tamil text and emoji?
6. Does the result change depending on the input? Avoid concluding that one tokenizer is universally better from a small test set.
The custom tokenizer's lowercasing behavior means it normalizes capitalization before encoding. BERT Base Uncased also uses uncased text processing. However, matching case behavior does not mean the two tokenizers will produce the same subwords or token counts.

### 5. Comparison results: fill in after running the experiment

The custom tokenizer's results are recorded above. The BERT comparison results should be copied from the executed comparison table rather than guessed.
| Comparison item | Custom Byte-Level BPE | BERT `bert-base-uncased` |
|---|---|---|
| Tokenization approach | Byte-level BPE; merges learned from this project's corpus | WordPiece with pretrained vocabulary |
| Vocabulary source | Learned by this implementation | Pretrained BERT vocabulary |
| Case handling | Lowercases input before encoding | Uncased BERT tokenizer |
| Initial units | 256 byte values | Pretrained WordPiece vocabulary and tokenizer rules |
| Token count for each test | Reported in this notebook | Run the comparison cell to record |
| Tamil and emoji behavior | Represented using UTF-8 bytes; inspect full-sequence behavior | Measure using the comparison cell |
| Exact original capitalization | Not preserved by normal encode()/decode() | Uncased tokenization does not preserve original capitalization in decoded normalized text |

Do not describe a tokenizer as better solely because it generates fewer tokens. Token count is one useful measurement, but quality also depends on the tokenizer's intended model, training data, language coverage, downstream task, and compatibility with that model.

# Part 4: Whitespace Tokenizer Baseline

Whitespace tokenization is a useful baseline because it is easy to implement and understand. It splits a string at whitespace boundaries, but usually leaves punctuation attached to neighboring words. For example, learning and learning! may be treated as different items.

To make a measured comparison, run a whitespace-tokenization baseline on the same comparison_texts list:
```python
whitespace_rows = []

for text in comparison_texts:
    whitespace_tokens = text.split()
    whitespace_rows.append({
        "input_text": text,
        "whitespace_tokens": whitespace_tokens,
        "whitespace_token_count": len(whitespace_tokens),
    })

whitespace_df = pd.DataFrame(whitespace_rows)
display(whitespace_df)
```

Whitespace counts are not directly equivalent to subword-token counts in every respect. This baseline is included to illustrate how tokenization rules affect the number and shape of text units.

# Part 5: Overall Analysis and Lessons Learned

This project demonstrates the main stages of building a tokenizer: text normalization, pre-tokenization, byte conversion, pair-frequency counting, iterative BPE merging, encoding, decoding, and saving/loading the learned vocabulary.

The custom implementation is valuable as a learning project because its merge rules and token IDs can be inspected directly. The pretrained BERT tokenizer provides a useful contrast: it uses a vocabulary and rules developed for an existing model rather than the merge rules learned in this experiment.

The current evidence supports the following conclusions:
- Frequent adjacent pairs can be merged into reusable subword tokens.
- A byte-level starting vocabulary can represent arbitrary UTF-8 byte sequences.
- Lowercasing makes capitalization variants share token IDs, but discards capitalization from the normal encoded representation.
- Vocabulary size, token count, and reconstruction checks measure different aspects of tokenizer behavior.
- A comparison with BERT should be based on the same test examples and actual recorded outputs.
- A tokenizer built independently is not a drop-in replacement for the tokenizer expected by an existing LLM. Its token IDs and vocabulary must match the model's training setup to be used with that model.

## Limitations and next steps

1. Run and save the AutoTokenizer comparison output.
2. Add a whitespace baseline and record its results.
3. Test a larger, clearly described held-out text set rather than relying only on a few examples.
4. Measure token counts and unknown-token behavior across English, Tamil, other scripts, numbers, punctuation, and emoji.
5. Improve exact text reconstruction if preserving capitalization and original formatting is a requirement.
6. Add automated tests and move the reusable tokenizer class into a separate Python module when the notebook experiments are stable.

## Conclusion

Building this tokenizer helped me understand how raw text becomes token IDs and how BPE learns subword units by merging frequent adjacent pairs. The notebook records successful training reconstruction, case-insensitive test checks, and save/reload verification for the custom implementation. Comparing it with whitespace tokenization and Hugging Face's pretrained BERT tokenizer will provide a clearer view of the trade-offs between simple word splitting, learned byte-level subwords, and a pretrained WordPiece vocabulary.
