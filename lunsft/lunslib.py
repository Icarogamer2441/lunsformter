import numpy as np
import re
import pickle
from collections import defaultdict, Counter
from .lunsformter import Lunsformter

class LunsDataset:
    def __init__(self, filepath=None, num_merges=1000):
        self.lines = []
        if filepath:  # allow empty dataset init (when loading)
            with open(filepath, 'r') as f:
                self.lines = [line.strip() for line in f if line.strip()]

            all_words = []
            for line in self.lines:
                words_in_line = re.sub(r'[,\.!\?]', '', line).split()
                all_words.extend(words_in_line)

            # Initialize tokens as list of chars + </w>
            corpus = []
            for word in all_words:
                chars = list(word) + ['</w>']
                corpus.append(chars)

            # Initialize vocabulary: each word as tuple
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
    def __init__(self, dataset, seq_len=12, dim=64, hidden_dim=128, num_layers=2, chunk_size=4, lr=0.03):
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

    def fit(self, epochs=500, verbose=True):
        for epoch in range(1, epochs+1):
            total_loss=0
            for tokens in self.dataset.data:
                if len(tokens)<2: continue
                inp = np.array(tokens[:-1])
                tgt = np.array(tokens[1:])
                logits = self.model.forward(inp)

                probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs /= np.sum(probs, axis=1, keepdims=True)

                onehot = np.zeros_like(probs)
                onehot[np.arange(len(tgt)), tgt] =1

                loss = -np.sum(np.log(probs[np.arange(len(tgt)),tgt]+1e-9))
                total_loss+=loss

                dlogits = (probs - onehot)/len(tgt)

                h = self.model.embeddings[inp] + self.model.positional[:len(inp)]
                gradWout = h.T @ dlogits
                gradbout = np.sum(dlogits, axis=0)
                self.model.output_W -= self.lr*gradWout
                self.model.output_b -= self.lr*gradbout

                dh = dlogits @ self.model.output_W.T
                self.model.embeddings[inp] -= self.lr*dh

            if verbose and (epoch%10==0 or epoch==1):
                print(f"Epoch {epoch}/{epochs} - Loss: {total_loss:.4f}")

    def generate(self, prompt, max_tokens=12, num_candidates=5, verbose=False, temperature=1.0):
        norm = re.sub(r'[,\.!\?]', '', prompt)
        bpe_tokens = []
        for word in norm.split():
            subwords = self.dataset.bpe_encode(word)
            ids = [self.dataset.word2idx.get(sw,1) for sw in subwords]
            bpe_tokens.extend(ids)
        seq = np.array(bpe_tokens)
        best_seq = self.model.insideout_generate(
            prefix=seq,
            max_tokens=max_tokens,
            num_candidates=num_candidates,
            penalize_repeats=True,
            verbose=verbose,
            temperature=temperature
        )
        safe_seq=[t if t>1 else 2 for t in best_seq]
        return self.dataset.detokenize(safe_seq)

    def save(self, path_prefix):
        np.savez(path_prefix+'_weights',
                 embeddings=self.model.embeddings,
                 positional=self.model.positional,
                 output_W=self.model.output_W,
                 output_b=self.model.output_b
                 )
        # Save variable-shaped layers separately with pickle
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
        # save tokenizer
        self.dataset.save_tokenizer(path_prefix+'_tokenizer.pkl')

    @classmethod
    def load(cls, path_prefix, dataset=None):
        data = np.load(path_prefix+'_weights.npz', allow_pickle=True)
        with open(path_prefix+'_config.pkl','rb') as f:
            cfg=pickle.load(f)

        # load heterogenous list of params from pickle
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
        # assign weights
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
            self.lr=lr
        # update dataset & vocab if needed
        self.dataset = new_dataset
        self.model.vocab_size = new_dataset.vocab_size
        self.fit(epochs=epochs, verbose=verbose)