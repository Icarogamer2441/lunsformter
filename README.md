# Lunsformter

A lightweight "transformer-inspired" language model with a novel **InsideOut** generation to explore multiple candidate sequences efficiently.

---

## Features

- Mini transformer-like model (`Lunsformter`)
- Byte-Pair Encoding tokenizer with vocabulary built from your text data
- Multi-candidate sampling (`insideout_generate`)
- Save, load, and resume training/fine-tuning models
- Fine-tune existing models with new datasets
- Simple training loop with numpy
- Easy integration via `LunsformterModel` wrapper

---

## Setup

- Requires **Python 3.7+**
- Uses `numpy`
- Project files:
  - `lunsft/lunsformter.py`: core model
  - `lunsft/lunslib.py`: wrapper, dataset, training, tokenizer, save/load
  - `examples/simple_lm.py`: sample script
  - `examples/chat_data.txt`: example dataset

---

## Usage

### 1. Prepare a text dataset

Put plain text lines in a `.txt` file. Example `examples/chat_data.txt`:

```
Hello my name is John
How are you today
Nice to meet you
```

---

### 2. Train & Save

Run the example script:

```bash
python examples/simple_lm.py
```

It will:

- train a model for **300 epochs**
- generate a sentence (before saving)
- save model & tokenizer to files (`examples/chat_model_*`)
- load the model back
- generate again from loaded model
- **fine-tune** loaded model for more epochs (e.g., 100)

---

### 3. Saving and Loading Your Own Model

After training, save your model:

```python
model.save('my_model_prefix')
```

Reload any time:

```python
loaded_model = LunsformterModel.load('my_model_prefix')
```

---

### 4. Fine-tuning Existing Models

Prepare a new dataset, then:

```python
new_dataset = LunsDataset('new_data.txt')
loaded_model.fine_tune(new_dataset, epochs=100)
```

You can then save it again if you'd like.

---

## Sample Code Snippet

```python
from lunsft import LunsDataset, LunsformterModel

# Training
dataset = LunsDataset('train_data.txt')
lm = LunsformterModel(dataset)
lm.fit(epochs=500)
lm.save('my_model')

# Later: load & generate
lm2 = LunsformterModel.load('my_model')
lm2.generate("Once upon a time", verbose=True)

# Fine-tune on new data
more_data = LunsDataset('more_text.txt')
lm2.fine_tune(more_data, epochs=100)
lm2.save('my_model_finetuned')
```

---

## Generation Function

| Parameter        | Description                                              | Default     |
|------------------|----------------------------------------------------------|-------------|
| `prompt`         | Input string                                             |             |
| `max_tokens`     | Max tokens to generate                                  | 12          |
| `num_candidates` | Number of initial top tokens to branch on               | 5           |
| `verbose`        | Print candidate step info during generation             | False       |

---

## InsideOut candidate generation

Instead of a single greedy decode:

1. Picks multiple top-N likely *start* tokens.
2. Expands each start token greedily to full sequences.
3. Scores based on average token likelihood.
4. Picks best overall candidate.

Use `verbose=True` to **see detailed candidate info**.

---

## License

MIT License

---

Enjoy experimenting and customizing **Lunsformter** for your needs!