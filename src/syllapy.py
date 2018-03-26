import collections
import os
import random

import numpy as np
import sklearn.model_selection
import sklearn.metrics
import torch
import torch.utils.data

random.seed(1585)
np.random.seed(1585)
torch.manual_seed(1585)


class LSTMTagger(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers,
                 embedding_dropout_p, vocab_size, tagset_size, batch_size):
        super(LSTMTagger, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.encoder = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = torch.nn.Dropout(p=embedding_dropout_p)
        self.lstm = torch.nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden(self.batch_size)

    def init_hidden(self, batch_size):
        return (torch.autograd.Variable(
                    torch.zeros(self.num_layers, batch_size, self.hidden_dim)),
                torch.autograd.Variable(
                    torch.zeros(self.num_layers, batch_size, self.hidden_dim)))
    
    def forward(self, sequence):
        embeddings = self.encoder(sequence)
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = torch.nn.functional.log_softmax(tag_space, dim=1)
        return tag_scores


class BiLSTMTagger(LSTMTagger):
    def __init__(self, embedding_dim, hidden_dim, num_layers,
                 embedding_dropout_p, vocab_size, tagset_size, batch_size):
        super(BiLSTMTagger, self).__init__(
            embedding_dim, hidden_dim, num_layers,
            embedding_dropout_p, vocab_size, tagset_size, batch_size
        )

        self.lstm = torch.nn.LSTM(
            embedding_dim, hidden_dim // 2, num_layers=num_layers,
            bidirectional=True, batch_first=True
        )

    def init_hidden(self, batch_size):
        return (torch.autograd.Variable(
                    torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2)),
                torch.autograd.Variable(
                    torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
        )
        


class SyllableData(torch.utils.data.Dataset):
    def __init__(self, fpath=None, data=None, transform=lambda x: x, shuffle_data=True):
        if data is None:
            self.data_samples = [line.strip() for line in open(fpath)]
        else:
            self.data_samples = data
        if shuffle_data:
            random.shuffle(self.data_samples)
        self.transform = transform

    def __len__(self):
        return len(self.data_samples)

    def __iter__(self):
        for sample in self.data_samples:
            yield self.transform(sample)

    def __getitem__(self, idx):
        return self.transform(self.data_samples[idx])

    def train_dev_test_split(self, dev_size=0.1, test_size=0.1):
        X_train, X_dev = sklearn.model_selection.train_test_split(self.data_samples, test_size=dev_size)
        X_train, X_test = sklearn.model_selection.train_test_split(X_train, test_size=test_size)
        return (SyllableData(data=X_train, transform=self.transform),
                SyllableData(data=X_dev, transform=self.transform),
                SyllableData(data=X_test, transform=self.transform))


class Sample2Tensor:
    def __init__(self, mark_word_boundaries=False, max_input_len=30):
        self.max_input_len = max_input_len
        self.mark_word_boundaries = mark_word_boundaries
        
        self.char2index = collections.defaultdict()
        self.char2index.default_factory = lambda: len(self.char2index)
        self.char2index['∂'] # padding char
        
    def __call__(self, sample):
        chars, tags = [], []
        sample = list(sample[:self.max_input_len])
        if self.mark_word_boundaries:
            sample = ['BoW'] + sample + ['EoW']
        for i, char in enumerate(sample):
            if char != '-':
                chars.append(char)
                if self.mark_word_boundaries:
                    tags.append(1 if i > 0 and sample[i - 1] == '-' else 0)
                else:
                    tags.append(1 if (i == 0) or (i > 0 and sample[i - 1] == '-') else 0)
        while len(chars) < (self.max_input_len):
            chars.append('∂')
            tags.append(0)
        return {'word': torch.LongTensor([self.char2index[char] for char in chars]),
                'tags': torch.LongTensor(tags)}


if __name__ == '__main__':
    batch_size = 300
    transformer = Sample2Tensor()
    data = SyllableData('../data/celex.txt', transform=transformer)
    train, dev, test = data.train_dev_test_split(dev_size=0.1, test_size=0.1)
    train_batches = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    dev_batches = torch.utils.data.DataLoader(dev, batch_size=batch_size, shuffle=True)
    test_batches = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
    tagger = BiLSTMTagger(40, 40, 1, 0.1, 100, 2, batch_size)
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.RMSprop(tagger.parameters(), lr=0.001)
    epochs = 5
    for epoch in range(epochs):
        epoch_loss = 0
        for i, batch in enumerate(train_batches):
            inputs = torch.autograd.Variable(batch['word'])
            targets =  torch.autograd.Variable(batch['tags'])
            tagger.zero_grad()            
            tagger.hidden = tagger.init_hidden(inputs.size(0))
            tag_scores = tagger(inputs)
            loss = loss_function(tag_scores.view(-1, tag_scores.size(2)), targets.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            if i > 0 and i % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i}/{len(data)/batch_size:.0f}], Loss: {epoch_loss.data[0] / i}\r', end='\r')
    #     dev_loss = 0
    #     for i, batch in enumerate(dev_batches):
    #         inputs = torch.autograd.Variable(batch['word'])
    #         targets =  torch.autograd.Variable(batch['tags'])
    #         tagger.zero_grad()            
    #         tagger.hidden = tagger.init_hidden()
    #         tag_scores = tagger(inputs)
    #         loss = loss_function(tag_scores.view(-1, tag_scores.size(2)), targets.view(-1))
    #         dev_loss += loss
    #     print(f"Dev loss {dev_loss.data[0] / len(dev_batches)}")
    # print(tag_scores)
    all_preds, all_true = [], []
    for i, batch in enumerate(test_batches):
        inputs = torch.autograd.Variable(batch['word'])
        targets = torch.autograd.Variable(batch['tags'])
        tagger.zero_grad()
        tagger.hidden = tagger.init_hidden(inputs.size(0))
        tag_scores = tagger(inputs)
        preds = tag_scores.view(-1, tag_scores.size(2)).data.numpy().argmax(1)
        y_true = targets.view(-1).data.numpy()
        all_preds.append(preds)
        all_true.append(y_true)
    print(sklearn.metrics.classification_report(np.hstack(all_true), np.hstack(all_preds), digits=4))

