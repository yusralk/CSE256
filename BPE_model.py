from collections import Counter
import torch
from torch.utils.data import Dataset

from sentiment_data import read_sentiment_examples

def merge_pair(symbols, pair):
    #method description: merge every occurrence of an adjacent pair in a sequence
    merged_symbols = []
    i = 0

    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            merged_symbols.append(symbols[i] + symbols[i+1])
            i += 2
        else:
            merged_symbols.append(symbols[i])
            i += 1

    return merged_symbols

def bpe_learn(sentences, target_vocab_size):
    #method description: learn BPE merges from training sentences
    #and return an ordered list of subword merges (learned_merges)
    #as well as the mapping of subwords to integer IDs (subword_to_idx)

    word_freqs = Counter(word for sentence in sentences for word in sentence.split())
    word_symbols = { word: list(word) + ["</W"] for word in word_freqs }

    #build initial vocab from individual symbols
    vocabulary = { symbol for symbols in word_symbols.values() for symbol in symbols}

    learned_merges = []

    while len(vocabulary) < target_vocab_size:
        pair_freqs = Counter()

        for word, symbols in word_symbols.items():
            frequency = word_freqs[word]

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                pair_freqs[pair] += frequency

        if not pair_freqs:
            break

        #only consider pairs that give new vocabulary symbols
        new_pairs = {
            pair: count for pair,count in pair_freqs.items()
            if pair[0] + pair[1] not in vocabulary
        }
        if not new_pairs:
            break

        #retrieve the most frequent pair
        strongest_pair = max(new_pairs, key=lambda vocab_pair: new_pairs[vocab_pair])
        merged_symbol = strongest_pair[0] + strongest_pair[1]
        learned_merges.append(strongest_pair)
        vocabulary.add(merged_symbol)

        #apply the merge to every word in training
        for word in word_symbols:
            word_symbols[word] = merge_pair(word_symbols[word], strongest_pair)

    #reserve 0 and 1 for UNKNOWN and Padding
    subword_to_idx = {
        symbol: idx + 2 for idx, symbol in enumerate(sorted(vocabulary))
    }

    return learned_merges, subword_to_idx

def tokenize_word(word, learned_merges):
    #description: split a word into subwords using the learned merge order
    symbols = list(word) + ["</W>"]

    for pair in learned_merges:
        symbols = merge_pair(symbols, pair)

    return symbols

class BPE_Dataset(Dataset):
    def __init__(self,
                 infile,
                 target_vocab_size = 5000,
                 learned_merges = None,
                 subword_to_idx = None):
        self.examples = read_sentiment_examples(infile)
        self.sentences = [" ".join(example.words) for example in self.examples]
        self.labels = [example.label for example in self.examples]

        if learned_merges is None:
            self.learned_merges, self.subword_to_idx = (
                bpe_learn(self.sentences, target_vocab_size))
        else:
            self.learned_merges = learned_merges
            self.subword_to_idx = subword_to_idx

        self.padding_idx = 0
        self.unk_idx = 1
        self.vocab_size = len(self.subword_to_idx) + 2

        self.token_indices = []

        for sentence in self.sentences:
            sentence_indices = []

            for word in sentence.split():
                subwords = tokenize_word(word, self.learned_merges)

                for subword in subwords:
                    idx = self.subword_to_idx.get(
                        subword, self.unk_idx
                    )
                    sentence_indices.append(idx)

            self.token_indices.append(sentence_indices)

        max_len = max(1, max(len(indices) for indices in self.token_indices))
        padded_sentences = []

        for indices in self.token_indices:
            padding_len = max_len - len(indices)
            padded_sentences.append(indices + [self.padding_idx] * padding_len)

        self.token_indices = torch.tensor(padded_sentences, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.token_indices[index], self.labels[index]







