# models.py

import torch
from torch import nn
import torch.nn.functional as F

from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset
import numpy as np

#use pretrained word embeddings
class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, embed_path):
        self.examples = read_sentiment_examples(infile)
        self.embed_path = embed_path
        self.word_embedding = read_word_embeddings(embed_path)

        #each sentence should be stored in a string separating terms with spaces
        self.sentences = [" ".join(example.words) for example in self.examples]
        self.labels = [example.label for example in self.examples]

        #find the average of the word vectors
        self.embeddings = []
        for s in self.sentences:
            words = s.split()
            embed_vector = [self.word_embedding.get_embedding(w) for w in words]
            self.embeddings.append(np.mean(embed_vector, axis=0))

        self.embeddings = np.array(self.embeddings)

        #converting the sentence vectors to a tensor for dataloader
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]

# Two-layer DAN model
class NN2DAN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)
        #self.dropout = nn.Dropout(0.6)
    def forward(self, x):
        #learn hidden features
        x = F.relu(self.fc1(x))
        #create scores for both classes
        #x = self.dropout(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x


# Three-layer DAN model
class NN3DAN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

        #self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        #first hidden layer
        x = F.relu(self.fc1(x))
        #second hidden layer
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        #find scores for both classes
        x = self.fc3(x)
        x = self.log_softmax(x)
        return x

#Randomly initialized embeddings
class SentimentDatasetDANRandom(Dataset):
    def __init__(self, infile, embed_path):
        self.examples = read_sentiment_examples(infile)
        self.word_embedding = read_word_embeddings(embed_path)

        #store the text and their sentiment labels
        self.sentences = [" ".join(example.words) for example in self.examples]
        self.labels = [example.label for example in self.examples]

        #find the longest sentence to standardize all examples to the same length
        sentence_length = max(len(ex.words) for ex in self.examples)
        self.token_indices = []

        for example in self.examples:
            indices = []
            #convert words to their vocab index
            for word in example.words:
                word_index = self.word_embedding.word_indexer.index_of(word)
                #if a word is UNKNOWN, use index 1
                if word_index == -1:
                    indices.append(1)
                else:
                    indices.append(word_index)

            #if somehow the sentence is longer than what the maximum we've found is, cut it off
            if len(indices) > sentence_length:
                indices = indices[:sentence_length]
            #pad shorter sentences with index 0
            elif len(indices) < sentence_length:
                indices = indices + [0 for _ in range(sentence_length-len(indices))]
            self.token_indices.append(indices)

        #embedding lookups need integers as indices
        self.token_indices = torch.tensor(self.token_indices, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.token_indices[index], self.labels[index]

#Two-layer DAN model with trainable & randomly initialized indices
class NN2DAN_RANDOM(nn.Module):
    def __init__(self, embed_size, hidden_size, embed_path):
        super().__init__()
        self.embed_path = embed_path
        word_embedding = read_word_embeddings(self.embed_path)

        #create a trainable vector for every vocab entry
        self.embedding = nn.Embedding(
            len(word_embedding.word_indexer),
            embed_size,
            padding_idx=0)

        self.dropout = nn.Dropout(0.6)
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, token_indices):
        #lookup word vectors
        token_indices = token_indices.long()
        word_vectors = self.embedding(token_indices)

        #ignore padding when finding word vector averages
        word_mask = (token_indices != 0).unsqueeze(-1)
        word_vectors = word_vectors * word_mask
        sentence_vectors = word_vectors.sum(dim=1)
        word_counts = word_mask.sum(dim=1).clamp(min=1)
        sentence_vectors = sentence_vectors/word_counts

        #predict the sentiment class
        x = F.relu(self.fc1(sentence_vectors))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


#Two-layer DAN model with trainable & randomly initialized indices
class NN3DAN_RANDOM(nn.Module):
    def __init__(self, embed_size, hidden_size, embed_path):
        super().__init__()
        self.embed_path = embed_path
        word_embedding = read_word_embeddings(self.embed_path)
        self.embedding = nn.Embedding(
            len(word_embedding.word_indexer),
            embed_size,
            padding_idx=0)

        self.dropout = nn.Dropout(0.65)
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)

    def forward(self, token_indices):
        # lookup word vectors
        token_indices = token_indices.long()
        word_vectors = self.embedding(token_indices)

        # ignore padding when finding word vector averages
        word_mask = (token_indices != 0).unsqueeze(-1)
        word_vectors = word_vectors * word_mask
        sentence_vectors = word_vectors.sum(dim=1)
        word_counts = word_mask.sum(dim=1).clamp(min=1)
        sentence_vectors = sentence_vectors / word_counts

        # predict the sentiment class
        x = F.relu(self.fc1(sentence_vectors))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

#Two-layer DAN model that uses Byte Pair Encoding
class NN2DAN_BPE(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()

        #create a trainable vector for every vocab entry
        self.embedding = nn.Embedding(
            vocab_size,
            embed_size,
            padding_idx=0)

        self.dropout = nn.Dropout(0.55)
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, token_indices):
        #lookup subword vectors
        token_indices = token_indices.long()
        word_vectors = self.embedding(token_indices)

        #ignore padding when finding word vector averages
        word_mask = (token_indices != 0).unsqueeze(-1)
        word_vectors = word_vectors * word_mask
        sentence_vectors = word_vectors.sum(dim=1)
        word_counts = word_mask.sum(dim=1).clamp(min=1)
        sentence_vectors = sentence_vectors/word_counts

        #predict the sentiment class
        x = F.relu(self.fc1(sentence_vectors))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


#Two-layer DAN model that uses Byte Pair Encoding
class NN3DAN_BPE(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size,
            padding_idx=0)

        self.dropout = nn.Dropout(0.65)
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)

    def forward(self, token_indices):
        # lookup BPE subword vectors
        token_indices = token_indices.long()
        word_vectors = self.embedding(token_indices)

        # ignore padding when finding word vector averages
        word_mask = (token_indices != 0).unsqueeze(-1)
        word_vectors = word_vectors * word_mask
        sentence_vectors = word_vectors.sum(dim=1)
        word_counts = word_mask.sum(dim=1).clamp(min=1)
        sentence_vectors = sentence_vectors / word_counts

        # predict the sentiment class
        x = F.relu(self.fc1(sentence_vectors))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
