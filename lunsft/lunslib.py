import numpy as np
import re
import pickle
from collections import defaultdict, Counter
import json
import csv
from .lunsformter import Lunsformter

def load_texts_from_source(filepath, file_format='txt', options=None):
    """
    Load list of strings from .txt, .csv, or .json file.

    Parameters:
    - filepath (str): path to the file
    - file_format (str): 'txt', 'csv', or 'json'
    - options (dict): format-specific options:
        * CSV: {'column': 'colname', 'delimiter': ',', 'quotechar': '"'}
        * JSON: {'json_path': 'root.subkey.list'}, supports nested keys

    Returns:
        list: list of text lines extracted from your file
    """
    if options is None:
        options = {}

    lines = []

    if file_format == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

    elif file_format == 'csv':
        column = options.get('column')
        delimiter = options.get('delimiter', ',')
        quotechar = options.get('quotechar', '"')
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter, quotechar=quotechar)
            for row in reader:
                if column and column in row:
                    text = row[column]
                    if text:
                        lines.append(text.strip())
                elif not column:
                    # take all columns' contents joined
                    texts = [str(v).strip() for v in row.values() if v]
                    combined = ' '.join(texts)
                    if combined:
                        lines.append(combined)

    elif file_format == 'json':
        json_path = options.get('json_path')  # support key1.key2.listidx syntax
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

            if json_path:
                parts = json_path.split('.')
                for p in parts:
                    if isinstance(data, list):
                        try:
                            idx = int(p)
                            data = data[idx]
                        except:
                            data = {}
                    elif isinstance(data, dict):
                        data = data.get(p, {})
                    else:
                        data = {}
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str) and item.strip():
                            lines.append(item.strip())
                        elif isinstance(item, dict):
                            texts = []
                            stack = [item]
                            while stack:
                                obj = stack.pop()
                                if isinstance(obj, dict):
                                    for v in obj.values():
                                        stack.append(v)
                                elif isinstance(obj, list):
                                    stack.extend(obj)
                                elif isinstance(obj, str):
                                    texts.append(obj.strip())
                            combined = ' '.join(texts)
                            if combined:
                                lines.append(combined)
                elif isinstance(data, str) and data.strip():
                    lines.append(data.strip())
            else:
                def extract_strings(obj):
                    if isinstance(obj, dict):
                        for v in obj.values():
                            yield from extract_strings(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            yield from extract_strings(item)
                    elif isinstance(obj, str):
                        yield obj.strip()
                combined_text = ' '.join(list(extract_strings(data)))
                if combined_text:
                    lines.append(combined_text)

    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    return lines

class ReadableResponseLayer:
    """
    A heuristic module to filter/choose the most readable/generated response.

    It generates multiple candidates, scores each
    based on average log-probability minus a repeat penalty,
    and selects the highest scoring candidate.

    Parameters:
    - threshold (float): minimum log-probability score to accept (default -3.5)
    - max_attempts (int): how many generation tries (default 5)
    - repetition_penalty (float): penalty factor multiplied to repeated tokens count (default 0.5)

    Usage:
        rlayer = ReadableResponseLayer(max_attempts=5)
        model = LunsformterModel(dataset, readable_layer=rlayer)
        text = model.generate('prompt')
    """
    def __init__(self, threshold=-3.5, max_attempts=5, repetition_penalty=0.5):
        self.threshold = threshold  # minimum average log-probability (log scale)
        self.max_attempts = max_attempts
        self.repetition_penalty = repetition_penalty

    def evaluate_score(self, tokens, prob_scores):
        avg_log_prob = np.mean(np.log(np.array(prob_scores) + 1e-12))  # avoid log(0)
        uniq, counts = np.unique(tokens, return_counts=True)
        repeats = sum(counts - 1)
        penalty = self.repetition_penalty * repeats / len(tokens)
        final_score = avg_log_prob - penalty
        return final_score

    def improve_or_accept(self, generate_func):
        best_tokens = None
        best_score = -np.inf
        for _ in range(self.max_attempts):
            tokens, scores = generate_func()
            score = self.evaluate_score(tokens, scores)
            if score > best_score:
                best_score = score
                best_tokens = tokens
        return best_tokens if best_tokens is not None else []

class LunsDataset:
    """
    Loads data, fits a BPE tokenizer, and encodes it as token sequences.

    Supports:
    - plain text files (one message / line)
    - CSV files, with optional column extraction
    - JSON files with nested path extraction

    Parameters:
    - filepath (str): dataset path
    - num_merges (int): number of Byte Pair Encoding merges (default 1000)
    - source_format (str): 'txt', 'csv', or 'json' (default 'txt')
    - options (dict): customize reading CSV/JSON (column, delimiter, json_path, etc.)

    Attributes:
    - data (list of list of int): list of token sequences
    - word2idx (dict): subword string to int id
    - idx2word (dict): int id to subword string
    - vocab_size (int)
    - bpe_merges (list of tuples): learned BPE merge steps

    Usage:

    ```python
    ds = LunsDataset('mydata.csv', source_format='csv', options={'column':'sentence'})
    tokens = ds.data[0]
    text = ds.detokenize(tokens)
    ```

    Methods:
    - `tokenize_line(text)` → list of ints
    - `detokenize(token_list)` → string
    - `save_tokenizer(path)`
    - `load_tokenizer(path)`
    """
    def __init__(self, filepath=None, num_merges=1000, source_format='txt', options=None):
        self.lines = []
        if filepath:
            if source_format in ['csv', 'json']:
                self.lines = load_texts_from_source(filepath, file_format=source_format, options=options)
            else:
                self.lines = load_texts_from_source(filepath, file_format='txt')

            all_words = []
            for line in self.lines:
                words_in_line = re.sub(r'[,\.!\?]', '', line).split()
                all_words.extend(words_in_line)

            corpus = []
            for word in all_words:
                chars = list(word) + ['</w>']
                corpus.append(chars)

            vocab = defaultdict(int)
            for word_token_list in corpus:
                vocab[tuple(word_token_list)] +=1

            merges = []
            for _ in range(num_merges):
                pairs = Counter()
                for word_tup, freq in vocab.items():
                    for i in range(len(word_tup)-1):
                        pairs[(word_tup[i], word_tup[i+1])] += freq
                if not pairs: break
                best = pairs.most_common(1)[0][0]
                merges.append(best)

                new_vocab = {}
                for word_tup, freq in vocab.items():
                    new_word = []
                    i = 0
                    while i < len(word_tup):
                        if i < len(word_tup)-1 and (word_tup[i], word_tup[i+1]) == best:
                            new_word.append(word_tup[i]+word_tup[i+1])
                            i +=2
                        else:
                            new_word.append(word_tup[i])
                            i +=1
                    new_vocab[tuple(new_word)] = new_vocab.get(tuple(new_word),0)+freq
                vocab = new_vocab

            self.bpe_merges = merges

            subwords = set()
            for word_tup in vocab:
                subwords.update(word_tup)

            self.word2idx = {'<pad>':0,'<unk>':1}
            self.idx2word = {0:'<pad>',1:'<unk>'}
            for idx,sw in enumerate(sorted(subwords), start=2):
                self.word2idx[sw]=idx
                self.idx2word[idx]=sw

            self.data=[]
            for line in self.lines:
                token_ids = self.tokenize_line(line)
                self.data.append(token_ids)

            self.vocab_size = len(self.word2idx)

    def bpe_encode(self,word):
        symbols = list(word)+['</w>']
        while True:
            pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
            merge_candidate=None
            for merge in self.bpe_merges:
                for i,pair in enumerate(pairs):
                    if pair==merge:
                        merge_candidate=(i,merge)
                        break
                if merge_candidate:
                    break
            if not merge_candidate: break
            i,_=merge_candidate
            symbols=symbols[:i]+[symbols[i]+symbols[i+1]]+symbols[i+2:]
        return symbols

    def tokenize_line(self,line):
        norm=re.sub(r'[,\.!\?]','',line)
        tokens=[]
        for word in norm.split():
            subwords = self.bpe_encode(word)
            tokens.extend([self.word2idx.get(sw,1) for sw in subwords])
        return tokens

    def detokenize(self,tokens):
        words=[]
        cw=''
        for idx in tokens:
            sw=self.idx2word.get(idx,'<unk>')
            if sw.endswith('</w>'):
                cw+=sw[:-4]
                words.append(cw)
                cw=''
            else:
                cw+=sw
        if cw: words.append(cw)
        return ' '.join(words)

    def save_tokenizer(self,fpath):
        data = {
            'bpe_merges': self.bpe_merges,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
        }
        with open(fpath,'wb') as f:
            pickle.dump(data,f)

    def load_tokenizer(self,fpath):
        with open(fpath,'rb') as f:
            data = pickle.load(f)
        self.bpe_merges = data['bpe_merges']
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.vocab_size = len(self.word2idx)

class LunsformterModel:
    """
    Compact transformer-like language model wrapper with inside-out generation and readable reranking.

    Parameters:
    - dataset (LunsDataset): pre-tokenized dataset instance
    - seq_len (int): default maximum sequence length (default 12)
    - dim (int): embedding dimension (default 64)
    - hidden_dim (int): hidden feedforward dimension (default 128)
    - num_layers (int): number of transformer layers (default 2)
    - chunk_size (int): chunk length for local attention (default 4)
    - lr (float): learning rate (default 0.03)
    - max_train_seq_len (int): max training sequence length, optional
    - readable_layer (ReadableResponseLayer): optional reranking for more coherent text

    Methods:

    - `fit(epochs, verbose, batch_size, shuffle)`
        Train model on the dataset for given epochs.
    
    - `generate(prompt, max_tokens, num_candidates, temperature, verbose)`
        Generate text extending a prompt using inside-out decoding.
    
    - `save(path_prefix)`
        Save model weights and tokenizer.

    - `load(path_prefix, dataset=None)`
        (classmethod). Re-create a saved model with weights and tokenizer.

    - `fine_tune(new_dataset, epochs, ...)`
        Further train/fine-tune on new data.

    Usage:

    ```python
    ds = LunsDataset('file.txt')
    model = LunsformterModel(ds, seq_len=20)
    model.fit(epochs=100)
    text = model.generate('Hello')
    model.save('model_path')
    ```
    """
    def __init__(self, dataset, seq_len=12, dim=64, hidden_dim=128, num_layers=2, chunk_size=4, lr=0.03, max_train_seq_len=None, readable_layer=None):
        self.dataset = dataset
        self.model = Lunsformter(
            vocab_size=dataset.vocab_size,
            seq_len=seq_len,
            dim=dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            chunk_size=chunk_size
        )
        self.lr = lr
        self.seq_len = seq_len
        self.max_train_seq_len = max_train_seq_len if max_train_seq_len is not None else seq_len
        self.readable_layer = readable_layer

    def fit(self, epochs=500, verbose=True, batch_size=1, shuffle=True):
        """
        Train your language model.

        Parameters:
        - epochs (int): Number of epochs to train (default 500)
        - verbose (bool): Print training status every 10 epochs (default True)
        - batch_size (int): How many samples per update (default 1)
        - shuffle (bool): Shuffle samples each epoch (default True)
        """
        N = len(self.dataset.data)
        for epoch in range(1, epochs + 1):
            total_loss = 0
            indices = np.arange(N)
            if shuffle:
                np.random.shuffle(indices)

            for start in range(0, N, batch_size):
                batch_idx = indices[start:start + batch_size]
                gradWout_accum = None
                gradbout_accum = None
                grad_embed_accum = {}
                batch_loss = 0

                for idx in batch_idx:
                    tokens = self.dataset.data[idx]
                    if len(tokens) < 2:
                        continue
                    inp = np.array(tokens[:-1])
                    tgt = np.array(tokens[1:])
                    if len(inp) > self.max_train_seq_len:
                        inp = inp[:self.max_train_seq_len]
                        tgt = tgt[:self.max_train_seq_len]
                    logits = self.model.forward(inp)

                    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                    probs /= np.sum(probs, axis=1, keepdims=True)

                    onehot = np.zeros_like(probs)
                    onehot[np.arange(len(tgt)), tgt] = 1

                    loss = -np.sum(np.log(probs[np.arange(len(tgt)), tgt] + 1e-9))
                    batch_loss += loss

                    dlogits = (probs - onehot) / len(tgt)

                    seq_len_inp = len(inp)
                    if seq_len_inp > self.model.positional.shape[0]:
                        extra_needed = seq_len_inp - self.model.positional.shape[0]
                        extra_pos = np.random.randn(extra_needed, self.model.dim) * 0.01
                        pos_embeds = np.concatenate([self.model.positional, extra_pos], axis=0)
                    else:
                        pos_embeds = self.model.positional

                    h = self.model.embeddings[inp] + pos_embeds[:seq_len_inp]

                    gradWout = h.T @ dlogits
                    gradbout = np.sum(dlogits, axis=0)
                    dh = dlogits @ self.model.output_W.T

                    if gradWout_accum is None:
                        gradWout_accum = gradWout
                        gradbout_accum = gradbout
                    else:
                        gradWout_accum += gradWout
                        gradbout_accum += gradbout

                    for posi, t in enumerate(inp):
                        if t not in grad_embed_accum:
                            grad_embed_accum[t] = np.zeros(self.model.embeddings.shape[1])
                        grad_embed_accum[t] += dh[posi]

                bs_actual = len(batch_idx)
                if gradWout_accum is not None and bs_actual > 0:
                    self.model.output_W -= self.lr * gradWout_accum / bs_actual
                    self.model.output_b -= self.lr * gradbout_accum / bs_actual

                    for t, grad_sum in grad_embed_accum.items():
                        self.model.embeddings[t] -= self.lr * grad_sum / bs_actual

                total_loss += batch_loss

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Epoch {epoch}/{epochs} - Loss: {total_loss:.4f}")

    def generate(self, prompt, max_tokens=12, num_candidates=5, verbose=False, temperature=1.0):
        norm = re.sub(r'[,\.!\?]', '', prompt)
        bpe_tokens = []
        for word in norm.split():
            subwords = self.dataset.bpe_encode(word)
            ids = [self.dataset.word2idx.get(sw,1) for sw in subwords]
            bpe_tokens.extend(ids)
        seq = np.array(bpe_tokens)

        readable_layer = self.readable_layer

        def generate_candidate():
            best_seq = self.model.insideout_generate(
                prefix=seq,
                max_tokens=max_tokens,
                num_candidates=num_candidates,
                penalize_repeats=True,
                verbose=verbose,
                temperature=temperature,
                return_scores=True
            )
            return best_seq

        if readable_layer is not None:
            best_tokens = readable_layer.improve_or_accept(generate_candidate)
        else:
            best_tokens, _ = generate_candidate()

        safe_seq=[t if t>1 else 2 for t in best_tokens]
        return self.dataset.detokenize(safe_seq)

    def save(self, path_prefix):
        np.savez(path_prefix+'_weights',
                 embeddings=self.model.embeddings,
                 positional=self.model.positional,
                 output_W=self.model.output_W,
                 output_b=self.model.output_b
                 )
        with open(path_prefix+'_layers.pkl', 'wb') as f:
            pickle.dump([ [Wg,bg,Ws,bs,Wo,bo] for ( (Wg,bg),(Ws,bs),(Wo,bo) ) in self.model.layers ], f)

        cfg = dict(
            seq_len=self.model.seq_len,
            dim = self.model.dim,
            hidden_dim = self.model.layers[0][1][0].shape[1] if self.model.layers else 128,
            num_layers=len(self.model.layers),
            chunk_size = self.model.chunk_size,
            vocab_size = self.model.vocab_size,
            lr = self.lr
        )
        with open(path_prefix+'_config.pkl','wb') as f:
            pickle.dump(cfg,f)
        self.dataset.save_tokenizer(path_prefix+'_tokenizer.pkl')

    @classmethod
    def load(cls, path_prefix, dataset=None):
        data = np.load(path_prefix+'_weights.npz', allow_pickle=True)
        with open(path_prefix+'_config.pkl','rb') as f:
            cfg=pickle.load(f)

        with open(path_prefix+'_layers.pkl','rb') as f:
            layer_params = pickle.load(f)

        if dataset is None:
            dataset = LunsDataset(None)
            dataset.load_tokenizer(path_prefix+'_tokenizer.pkl')
            dataset.data = []
        else:
            dataset.load_tokenizer(path_prefix+'_tokenizer.pkl')

        lm = cls(dataset,
                 seq_len=cfg['seq_len'],
                 dim=cfg['dim'],
                 hidden_dim=cfg['hidden_dim'],
                 num_layers=cfg['num_layers'],
                 chunk_size=cfg['chunk_size'],
                 lr=cfg.get('lr',0.03)
                 )
        lm.model.embeddings = data['embeddings']
        lm.model.positional = data['positional']
        lm.model.output_W = data['output_W']
        lm.model.output_b = data['output_b']
        lm.model.layers = []
        for Wg,bg,Ws,bs,Wo,bo in layer_params:
            gate = (Wg,bg)
            sculpt = (Ws,bs)
            out = (Wo,bo)
            lm.model.layers.append( (gate,sculpt,out) )
        return lm

    def fine_tune(self, new_dataset, epochs=100, verbose=True, lr=None):
        if lr is not None:
            self.lr = lr
        self.dataset = new_dataset
        self.model.vocab_size = new_dataset.vocab_size
        self.fit(epochs=epochs, verbose=verbose)