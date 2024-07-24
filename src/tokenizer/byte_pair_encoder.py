import collections
import re

from tqdm import tqdm


class BytePairEncoder:
    def __init__(self):
        self.merges = None
        self.characters = set()
        self.tokens = collections.Counter()
        self.vocab = {}
        self.space_token = "_"

    def split_into_sentences(self, text):
        """Splits text into sentences using punctuation and line breaks."""
        pattern = re.compile(r"(?<=\.)|(?<=。)|(?<=．)|(?<=\n)")
        sentences = pattern.split(text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def simple_word_tokenize(self, text: str):
        """Tokenizes text into words and punctuation."""
        return re.findall(r"\b\w+\b|[^\w\s]", text)

    def normalize_text(self, text):
        """Normalizes text by converting to lowercase and collapsing whitespace."""
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def split_text_into_words(self, text):
        """Splits text into words, prefixing non-first words with a special token."""
        return [
            self.space_token + word if idx != 0 else word
            for idx, word in enumerate(text.split())
        ]

    def update_vocab(self, words):
        """Updates the vocabulary with the words from the text."""
        for word in words:
            tokenized_word = " ".join(list(word))
            self.vocab[tokenized_word] = self.vocab.get(tokenized_word, 0) + 1
        self.tokens.update("".join(words))

    def initialize_vocab(self, text):
        """Initializes the vocabulary from the given text."""
        text = self.normalize_text(text)
        words = self.split_text_into_words(text)
        self.update_vocab(words)

    def get_bigram_counts(self):
        """Counts the frequency of each bigram in the vocabulary."""
        pairs = collections.Counter()
        for word, count in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += count
        return pairs

    def merge_vocab(self, pair):
        """Merges the most frequent pair in the vocabulary."""
        vocab_out = {}
        bigram = " ".join(pair)
        bigram_regex = re.escape(bigram)
        p = re.compile(rf"(?<!\S){bigram_regex}(?!\S)")
        merged_symbol = "".join(pair)

        for word in self.vocab:
            merged_word = p.sub(merged_symbol, word)
            vocab_out[merged_word] = self.vocab[word]

        self.vocab = vocab_out
        return merged_symbol

    def perform_merges(self, num_merges):
        """Performs a specified number of merges to learn subword tokens."""
        for _ in tqdm(range(num_merges), desc="Merging"):
            pairs = self.get_bigram_counts()
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            best_count = pairs[best_pair]
            merged_symbol = self.merge_vocab(best_pair)
            self.tokens[merged_symbol] = best_count

    def train(self, text, num_merges):
        """Trains the BPE model on the provided text."""
        sentences = self.split_into_sentences(text)
        for sentence in sentences:
            self.initialize_vocab(sentence)
        self.characters = set(self.tokens.keys())
        self.perform_merges(num_merges)

    def tokenize(self, text):
        """Tokenizes text using the trained BPE model."""
        text = self.normalize_text(text)
        words = self.split_text_into_words(text)
        output_tokens = []

        for word in words:
            word = "".join(list(word))
            while word:
                for i in range(len(word), 0, -1):
                    subword = word[:i]
                    if subword in self.tokens:
                        output_tokens.append(subword)
                        word = word[i:]
                        break
                else:
                    # If no known subword is found, treat the character as unknown
                    output_tokens.append(word[0])
                    word = word[1:]

        return output_tokens
