# Lunsformter

A minimal, lightweight research-oriented **Inside-Out Chunk-based Transformer** with **BPE tokenizer** implemented in **NumPy**.

Designed for experiments on:

- chunk-based local/global context propagation
- gating mechanisms
- inside-out generation decoding
- fast BPE tokenization

---

## Features

- Byte Pair Encoding (BPE) tokenizer with customizable merge ops.
- Chunked Transformer with chunk linking and sculpting layers.
- Inside-out generation with multiple candidates.
- Save/load models including tokenizer.
- Minimal test & example script.

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

# Or use
pip install .
```

---

## Usage Example

### Dataset

Prepare a simple text file, e.g. `examples/chat_data.txt`:

```
Hello how are you
I am fine thank you
What's your name
My name is Lunsy
Nice to meet you
...
```

### Train and generate

Run our minimal test script:

```bash
python run_train_and_generate.py
```

This performs:

- Tokenizes your file.
- Trains a model.
- Saves it.
- Loads saved model.
- Generates samples.
- Fine-tunes & generates again.

---

### API Overview

```python
from lunsft.lunslib import LunsDataset, LunsformterModel

ds = LunsDataset(filepath='examples/chat_data.txt', num_merges=200)

model = LunsformterModel(ds, seq_len=12, dim=64, hidden_dim=128, num_layers=2, chunk_size=4)

model.fit(epochs=100)

print(model.generate("Hello I am", max_tokens=20))

model.save('mymodel/lunsy')

reloaded = LunsformterModel.load('mymodel/lunsy')

print(reloaded.generate("Hello again", max_tokens=20))
```

----

## File Structure

```
run_train_and_generate.py    # Minimal example script
setup.py                     # Packaging
requirements.txt
README.md

lunsft/
    __init__.py
    lunslib.py               # Dataset, training, API
    lunsformter.py           # Model class, inside-out decoding

examples/
    chat_data.txt            # Sample data file
    simple_lm.py             # Optional older LM example
```

----

## License

MIT License.

---

## Contributions

Feel free to fork and experiment, PRs are welcome!

------

Enjoy!