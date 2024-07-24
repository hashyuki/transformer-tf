import collections
import re

import numpy as np
from scipy.special import digamma


class Unigram:
    def __init__(self):
        self.maxlen = None
        self.vocab_size = None
        self.subword_logp = None

    def split_into_sentences(self, text):
        """Splits the text into sentences based on punctuation and line breaks."""
        pattern = re.compile(r"(?<=\.)|(?<=。)|(?<=．)|(?<=\n)")
        sentences = pattern.split(text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def normalize_text(self, text):
        """Normalizes the text by converting to lowercase and collapsing whitespace."""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def forward_step(self, text, trie):
        """Performs the forward step of the Viterbi algorithm to find the best tokenization."""
        N = len(text)
        d = [-np.inf] * (N + 1)  # Maximum log-prob of any tokenization of text[:i]
        p = [None] * (N + 1)  # Tracks the number of characters of the final token
        d[0] = 0

        for i in range(1, N + 1):
            for j in range(max(i - self.maxlen, 0), i):
                final_token = text[j:i]
                final_value = trie.get_value(final_token)
                if final_value and d[j] + final_value > d[i]:
                    d[i] = d[j] + final_value
                    p[i] = len(final_token)

            if p[i] is None:
                raise ValueError(f"Encountered unknown token '{text[i-1]}'.")

        return d[-1], p

    def backward_step(self, text, p):
        """Reconstructs the tokenization from the parent pointers."""
        idx = len(p)
        tokenization = []
        while idx > 1:
            next_idx = idx - p[idx - 1]
            token = text[next_idx - 1 : idx - 1]
            tokenization.append(token)
            idx = next_idx
        tokenization.reverse()
        return tokenization

    def viterbi_forward(self, word, subword_logp):
        """Forward pass of the Viterbi algorithm for subword segmentation."""
        best_subw_slices = [None] * (len(word) + 1)
        neg_loglik = np.zeros(len(word) + 1)
        neg_loglik[0] = 0

        for eow in range(1, len(word) + 1):
            neg_loglik[eow] = np.inf
            for bow in range(eow):
                subw = word[bow:eow]
                if subw in subword_logp:
                    logp = subword_logp[subw]
                    score = neg_loglik[bow] - logp
                    if score < neg_loglik[eow]:
                        neg_loglik[eow] = score
                        best_subw_slices[eow] = (bow, eow)
        return neg_loglik, best_subw_slices

    def viterbi_backward(self, word, subw_slices):
        """Backward pass to reconstruct the subwords from the best slices."""
        subwords = []
        next_slices = subw_slices[-1]
        while next_slices is not None:
            subw = word[next_slices[0] : next_slices[1]]
            subwords.append(subw)
            next_slices = subw_slices[next_slices[0]]
        subwords.reverse()
        return subwords

    def get_viterbi_path(self, word, subword_logp):
        """Gets the most likely subword tokenization and its loss."""
        neg_loglik, best_subw_slices = self.viterbi_forward(word, subword_logp)
        subwords = self.viterbi_backward(word, best_subw_slices)
        vit_path_loss = neg_loglik[-1]
        return subwords, vit_path_loss

    def E_step(self, tokenization, subword_logp):
        """Expectation step: update subword log-probabilities."""
        counts = collections.Counter(tokenization)
        norm = sum(counts.values())

        logsum = digamma(norm)
        for k, v in counts.items():
            counts[k] = digamma(v) - logsum

        for k, v in counts.items():
            subword_logp[k] = v
        return subword_logp

    def M_step(self, text, subword_logp):
        """Maximization step: find the best tokenization and calculate the total loss."""
        viterbi_subword_freq = collections.Counter()
        vit_path_loss_full = 0

        sentences = self.split_into_sentences(text)
        for sentence in sentences:
            sentence = self.normalize_text(sentence)
            subwords, vit_path_loss = self.get_viterbi_path(sentence, subword_logp)
            vit_path_loss_full += vit_path_loss
            viterbi_subword_freq.update(subwords)

        return list(viterbi_subword_freq.keys()), vit_path_loss_full

    def EM_step(self, text, tokenization, subword_logp):
        """Performs a single EM step, including both E-step and M-step."""
        subword_logp = self.E_step(tokenization, subword_logp)
        tokenization, loss = self.M_step(text, subword_logp)
        return loss, tokenization, subword_logp

    def EM_round(self, text, tokens, delta=0.01, max_iter=10):
        """Performs multiple EM iterations to refine the model."""
        total_count = sum(tokens.values())
        self.subword_logp = {k: np.log(v / total_count) for k, v in tokens.items()}
        tokenization, old_loss = self.M_step(text, self.subword_logp)

        for step in range(max_iter):
            print(f"EM iter {step}: ", end="")
            loss, tokenization, subword_logp = self.EM_step(
                text, tokenization, self.subword_logp
            )
            print(f"Loss={loss:.2f}")
            if abs(old_loss - loss) < delta:
                break
            old_loss = loss

    def prune_tokens(self, tokens, characters, vocab_size, trim_frac=0.2):
        """Prunes tokens to reduce the vocabulary size."""
        sorted_tokens = tokens.most_common()
        num_tokens = len(sorted_tokens)
        n_trim = int(trim_frac * num_tokens)

        for i in reversed(range(num_tokens)):
            if num_tokens <= vocab_size:
                return False
            if n_trim <= 0:
                return True
            token = sorted_tokens[i][0]
            if token not in characters:
                self.subword_logp[token] = 0
                tokens.pop(token)
                n_trim -= 1
                num_tokens -= 1

        if n_trim > 0:
            raise ValueError(
                "Could not reduce tokens further. Please increase vocab size."
            )
        return False

    def train(
        self, text, tokens, characters, vocab_size, delta=0.01, max_iter=5, max_rounds=5
    ):
        """Trains the model using the EM algorithm and pruning."""
        if vocab_size > len(tokens):
            raise ValueError(
                f"Vocab size is larger than the available number of tokens {len(tokens)}."
            )

        for i in range(1, max_rounds + 1):
            print(f"--- Round {i}. Vocab size: {len(tokens)} ---")
            self.EM_round(text, tokens, delta, max_iter)
            if not self.prune_tokens(tokens, characters, vocab_size):
                break
        self.vocab_size = len(tokens)

    def tokenize(self, text):
        """Tokenizes the text using the trained model."""
        text = text.lower()
        text = re.sub(" ", "_", text)
        if self.subword_logp is None:
            raise ValueError("Trainer has not yet been fit. Cannot tokenize.")
        tokens, _ = self.get_viterbi_path(text, self.subword_logp)
        return tokens
