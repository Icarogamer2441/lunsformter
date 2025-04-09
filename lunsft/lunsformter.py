import numpy as np

class Lunsformter:
    def __init__(self, vocab_size=50, seq_len=10, dim=64, hidden_dim=128, num_layers=2, chunk_size=5):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dim = dim
        self.chunk_size = chunk_size

        self.embeddings = np.random.randn(vocab_size, dim) * 0.01
        self.positional = np.random.randn(seq_len, dim) * 0.01

        # Layer parameters: lists of (WX, bX)
        self.layers = []
        for _ in range(num_layers):
            W_g = np.random.randn(dim, dim) / np.sqrt(dim)    # Gating lens
            b_g = np.zeros(dim)
            W_s = np.random.randn(dim, hidden_dim) / np.sqrt(dim) # Sculpting
            b_s = np.zeros(hidden_dim)
            W_o = np.random.randn(hidden_dim, dim) / np.sqrt(hidden_dim)  # Back to dim
            b_o = np.zeros(dim)
            self.layers.append(((W_g, b_g), (W_s, b_s), (W_o, b_o)))

        # Output projection
        self.output_W = np.random.randn(dim, vocab_size) / np.sqrt(dim)
        self.output_b = np.zeros(vocab_size)

    def _lensed_gate(self, x, W_g, b_g):
        gate = 1 / (1 + np.exp(-(x @ W_g + b_g)))
        return x * gate

    def _context_sculpt(self, x, W_s, b_s, W_o, b_o):
        sculpted = np.tanh(x @ W_s + b_s)
        out = sculpted @ W_o + b_o
        return out

    def _chunk_link_update(self, x):
        seq_len, dim = x.shape
        rem = seq_len % self.chunk_size
        if rem != 0:
            pad_len = self.chunk_size - rem
            pad = np.zeros((pad_len, dim), dtype=x.dtype)
            x_padded = np.concatenate([x, pad], axis=0)
        else:
            x_padded = x

        chunks = x_padded.reshape(-1, self.chunk_size, self.dim)

        new_chunks = []
        for i, c in enumerate(chunks):
            neighbor_sum = c.copy()
            if i > 0:
                neighbor_sum += chunks[i-1] * 0.5
            if i < len(chunks) - 1:
                neighbor_sum += chunks[i+1] * 0.5
            updated = neighbor_sum / (1 + 0.5 * (i > 0) + 0.5 * (i < len(chunks) -1))
            new_chunks.append(updated)

        updated_x = np.concatenate(new_chunks, axis=0)

        # Remove padding
        return updated_x[:seq_len]

    def forward(self, idx_seq):
        seq_len_in = len(idx_seq)
        if seq_len_in > self.positional.shape[0]:
            extra_needed = seq_len_in - self.positional.shape[0]
            extra_pos = np.random.randn(extra_needed, self.dim) * 0.01
            pos_embeds = np.concatenate([self.positional, extra_pos], axis=0)
        else:
            pos_embeds = self.positional

        x = self.embeddings[idx_seq] + pos_embeds[:seq_len_in]

        for pos in range(1, len(x)):
            decay = 0.8 ** pos
            x[pos] += decay * x[0]

        for (gate_params, sculpt_params, out_params) in self.layers:
            W_g, b_g = gate_params
            W_s, b_s = sculpt_params
            W_o, b_o = out_params

            residual = x
            x = self._lensed_gate(x, W_g, b_g)
            x = self._context_sculpt(x, W_s, b_s, W_o, b_o)
            x += residual
            x = self._chunk_link_update(x)

        logits = x @ self.output_W + self.output_b
        return logits

    def generate(self, prefix, max_tokens=20, temperature=1.0, return_scores=False):
        idx_seq = np.array(prefix)
        prob_scores = []
        for _ in range(max_tokens):
            logits = self.forward(idx_seq[-self.seq_len:])
            logits = logits[-1] / max(temperature, 1e-8)
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)
            next_token = np.random.choice(len(probs), p=probs)
            prob_scores.append(probs[next_token])
            idx_seq = np.append(idx_seq, next_token)
        if return_scores:
            return idx_seq, prob_scores
        else:
            return idx_seq

    def insideout_generate(self, prefix, max_tokens=20, num_candidates=5, penalize_repeats=True, verbose=False, temperature=1.0, return_scores=False):
        idx_seq = np.array(prefix)

        logits = self.forward(idx_seq[-self.seq_len:])
        logits = logits[-1] / max(temperature, 1e-8)
        start_probs = np.exp(logits - np.max(logits))
        start_probs /= np.sum(start_probs)
        top_tokens = np.argpartition(-start_probs, num_candidates)[:num_candidates]

        if verbose:
            print(f"[InsideOut] Top {num_candidates} initial tokens to try: {top_tokens}")

        best_seq = None
        best_score = -np.inf
        best_scores_list = None

        for candidate_idx, start in enumerate(top_tokens):
            candidate_seq = np.append(idx_seq, start)
            scores = [start_probs[start]]

            if verbose:
                print(f"\n[InsideOut] Candidate {candidate_idx +1}/{len(top_tokens)} starting token {start}")

            for step in range(max_tokens -1):
                logits_cand = self.forward(candidate_seq[-self.seq_len:])
                logits_cand = logits_cand[-1] / max(temperature, 1e-8)
                next_probs = np.exp(logits_cand - np.max(logits_cand))
                next_probs /= np.sum(next_probs)

                next_token = np.random.choice(len(next_probs), p=next_probs)

                repeat_penalty = 0.5 if penalize_repeats and next_token in candidate_seq else 1.0
                prob_score = next_probs[next_token] * repeat_penalty
                scores.append(prob_score)

                candidate_seq = np.append(candidate_seq, next_token)

                if verbose:
                    print(f"  Step {step +1}/{max_tokens-1} token: {next_token}, prob_score: {prob_score:.4f}")

            avg_score = np.mean(scores)
            if verbose:
                print(f"  Candidate avg score: {avg_score:.4f}")

            if avg_score > best_score:
                best_score = avg_score
                best_seq = candidate_seq
                best_scores_list = list(scores)
                if verbose:
                    print(f"  --> New best sequence found with score {best_score:.4f}")

        if verbose:
            print(f"\n[InsideOut] Best overall avg score: {best_score:.4f}")

        if return_scores:
            return best_seq, best_scores_list
        else:
            return best_seq

if __name__ == "__main__":
    model = Lunsformter()
    prefix = [1, 2, 3]

    generated = model.generate(prefix, max_tokens=10)
    print("Generated Token IDs (standard generate):", generated)

    print("\nStarting verbose insideout_generate...")
    insideout_verbose = model.insideout_generate(prefix, max_tokens=10, num_candidates=3, verbose=True)
    print("InsideOut Generated Token IDs (verbose):", insideout_verbose)

    print("\nInsideout_generate without verbose...")
    insideout_silent = model.insideout_generate(prefix, max_tokens=10, num_candidates=3, verbose=False)
    print("InsideOut Generated Token IDs (silent):", insideout_silent)