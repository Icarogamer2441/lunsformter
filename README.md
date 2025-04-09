# Lunsformter

A minimal, lightweight research-oriented **Inside-Out Chunk-based Transformer** with **BPE tokenizer** implemented in **NumPy**.

Designed for experiments on:

- chunk-based local/global context propagation
- gating mechanisms
- inside-out generation decoding
- fast BPE tokenization

---

## Table of Contents

- [Features](#features)
- [Supported Data Formats](#supported-data-formats)
- [Installation](#installation)
- [Usage Example](#usage-example)
- [API Overview](#api-overview)
- [Advanced Features](#advanced-features)
- [File Structure](#file-structure)
- [License](#license)
- [Contributions](#contributions)

---

## Features

- **Flexible dataset loading**: Supports plain TXT, CSV (choose column), and JSON (specify nested path)
- **BPE subword tokenizer** with customizable merge operations
- **Compact chunk-based Transformer** with chunk linking and sculpting layers
- **Inside-out generation decoding** (expands tokens outward around prompt) with multiple candidates & optional penalties
- **Readable Response Layer**: An optional smart reranker that filters out low-quality generations, improving coherence
- **Tiny, minimal pure NumPy implementation** suitable for experiments & tutorials
- **Customizable training**: Control epochs, layers, dims, sequence length, learning rate
- **Save/load models with tokenizer** included
- **Minimal train & generate sample script** included (`train_and_generate_sample.py`)
- **Zero deep learning framework dependencies** (no TensorFlow, Torch needed!)
- Open source, MIT license

---

## Supported Data Formats

### Plain Text (`.txt`)

A simple text file, one message per line:

```
Hello how are you
I am good thanks
Nice to meet you
What's your plan today?
```

### CSV File (`.csv`)

Example:

| user  | message                  | sentiment |
|--------|---------------------------|-----------|
| bob    | Hello, how are you?       | positive  |
| alice  | I'm fine, thanks!         | positive  |

You can select a column:

```python
dataset = LunsDataset('chat.csv', source_format='csv', options={'column': 'message'})
```

### JSON File (`.json`)

Example:

```json
{
  "conversations": [
    {"text": "Hello there!"},
    {"text": "How may I assist you?"},
    {"text": "Thank you very much!"}
  ]
}
```

Select nested path:

```python
dataset = LunsDataset('chat.json',
                      source_format='json',
                      options={'json_path': 'conversations'})
```

or for nested lists or sub-fields e.g., `'conversations'` → list of dicts → will join string fields inside each dict.

---

## Installation

Recommended to use **Python 3.8+**.

```bash
# Download this repo
git clone https://github.com/Icarogamer2441/lunsformter.git
cd lunsformter

# (Optional) create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quickstart

### 1. Prepare your dataset

- As plain `.txt`, or  
- `.csv` with a message/content column, or  
- `.json` with nested list/path of strings

Save in `examples/chat_data.txt` or another file.

### 2. Train & generate sample

Run the provided full example script:

```bash
python train_and_generate_sample.py
```

This will:

- Load dataset (`examples/chat_data.txt`)
- Train a tiny model (~100 epochs)
- Save/load during process
- Generate a reply with `"Hello"` prompt

### 3. Minimal code snippet

Alternatively, do a quick training in Python:

```python
from lunsft.lunslib import LunsDataset, LunsformterModel

dataset = LunsDataset('examples/chat_data.txt')

model = LunsformterModel(dataset)
model.fit(epochs=100)

print(model.generate('Hello'))
```

---

## Usage Example

### Prepare Dataset

Plain text (default) or specify CSV or JSON:

```python
dataset = LunsDataset('myfile.txt')
# or CSV column:
dataset = LunsDataset('data.csv', source_format='csv', options={'column': 'content'})
# or nested JSON path:
dataset = LunsDataset('data.json', source_format='json', options={'json_path': 'records.messages'})
```

### Initialize & train model

```python
model = LunsformterModel(dataset,
                         seq_len=20,
                         dim=64,
                         hidden_dim=128,
                         num_layers=2,
                         lr=0.02)
model.fit(epochs=100)
```

### Generate text

```python
output = model.generate('Hello', max_tokens=20, temperature=0.9)
print(output)
```

### Save & load later

```python
model.save('saved_model')  # Saves to files with this prefix
loaded_model = LunsformterModel.load('saved_model')
```

---

## API Overview

- `LunsDataset(filepath, source_format, options)`: loads & tokenizes dataset, supports text/CSV/JSON
- `LunsformterModel(dataset, seq_len, dim, hidden_dim, num_layers, lr, ...)` initializes a model
- `.fit(epochs=100)`: trains model
- `.generate(prompt, max_tokens=N, temperature=P)`: generate continuation text
- `.save('pathprefix')` and `LunsformterModel.load('pathprefix')`: persist & restore

Parameters like chunk_size, learning rate (`lr`), repetitions, Readable Layer config can also be customized for better results.

---

## Advanced Feature: Readable Response Layer

Optionally, improve generation quality with a **quality-checker heuristic layer**.

This `ReadableResponseLayer`:

- reruns generation multiple times
- scores each candidate based on average log-probability and repetition
- accepts the best/highest scoring version

### How to use

```python
from lunsft.lunslib import ReadableResponseLayer

rlayer = ReadableResponseLayer(threshold=-3.5, max_attempts=5, repetition_penalty=0.5)

# Provide at init (recommended)
model = LunsformterModel(dataset, readable_layer=rlayer)

# Or: attach after model creation
model.readable_layer = rlayer

# Then generate as usual
reply = model.generate("Hello!", max_tokens=20)
print(reply)
```

### Parameters

- `threshold`: minimum score for acceptance (default -3.5)
- `max_attempts`: number of retries (default 5)
- `repetition_penalty`: discourage repeated tokens (default 0.5)

Tune these to control strictness & diversity.

---

This reranking step can improve **coherence** and **fluency** in outputs — but is **optional** and adds some runtime cost.

---

## File Structure

```
.
├── README.md               # Full library documentation & usage
├── requirements.txt        # Package dependencies
├── setup.py                # Python package installer script
├── run_train_and_generate.py   # Minimal example script
├── train_and_generate_sample.py # **Ready-to-run sample: load, train, save, load, generate**

├── lunsft/                 # Core library source code
│   ├── __init__.py
│   ├── lunslib.py          # Dataset loaders, BPE tokenizer, training logic, API wrappers
│   ├── lunsformter.py      # Transformer-like model & inside-out generation

├── examples/               # Example datasets & demos
│   ├── chat_data.txt       # Small example dialogue corpus (plain text)
│   ├── simple_lm.py        # Alternative old-style LM example

```

---

## License

This project is licensed under the **MIT License** — free for research, experiments, and commercial purposes. See [LICENSE](LICENSE) file for details.

---

## Status & Contributions

- **Status:** Experimental, educational, and for prototyping  
- **Not designed** for production deployment

---

### Contributions

- Issues, ideas, and pull requests welcome!  
- Fork and experiment freely  
- Add examples, datasets, improvements :)

---

## Enjoy!

Happy tinkering, training, and exploring! 🚀

---