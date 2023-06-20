import torch
import os
import gensim
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import defaultdict
from config import *


def seed_evetything(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_evetything(3847)

logger = open(LOG_PATH, "w")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str):
    print(msg)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    logger.write("{}|{}\n".format(time_str, msg))
    logger.flush()


def get_word_count():
    word_count = defaultdict(int)
    for filename in [TRAIN_SET_PATH, VALIDATION_SET_PATH]:
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                line = line.split()[1:]
                for word in line:
                    word_count[word] += 1
    return word_count


def build_vocab():
    word_count = get_word_count()
    word2idx = defaultdict(lambda: 1)
    word2idx["<PAD>"] = 0
    word2idx["<UNK>"] = 1
    word2idx["<START>"] = 2
    word2idx["<EOP>"] = 3

    for filename in [TRAIN_SET_PATH, VALIDATION_SET_PATH]:
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                line = line.split()[1:]
                for word in line:
                    if word not in word2idx and word_count[word] >= 20:
                        word2idx[word] = len(word2idx)
    return word2idx


def build_embedding_matrix(word2idx):
    if os.path.exists(EMBEDDING_MATRIX_PATH):
        return np.load(EMBEDDING_MATRIX_PATH)
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(
        WORD2VEC_PATH, binary=True
    )
    embedding_matrix = np.random.uniform(-1, 1, [len(word2idx), word2vec.vector_size])
    for word, idx in word2idx.items():
        if word in word2vec:
            embedding_matrix[idx] = word2vec[word]
    np.save(EMBEDDING_MATRIX_PATH, embedding_matrix)
    return embedding_matrix


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, path, word2ix, seq_len=100, is_train=False):
        super(TextDataset, self).__init__()
        self.seq_len = seq_len
        self.word2ix = word2ix
        self.is_train = is_train
        self.data, self.label = self.load_data(path)

    def process_line(self, data):
        if len(data) > self.seq_len:
            data = data[: self.seq_len]
        elif len(data) < self.seq_len:
            data += ["<PAD>"] * (self.seq_len - len(data))
        data = ["<START>"] + data + ["<EOP>"]
        data = [self.word2ix[word] for word in data]
        return data

    def load_data(self, path):
        X, y = [], []
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                line = line.split()
                data = line[1:]
                label = int(line[0])

                if self.is_train:
                    while len(data):
                        x = self.process_line(data)
                        X.append(x)
                        y.append(label)
                        data = data[self.seq_len:]
                else:
                    data = self.process_line(data)
                    X.append(data)
                    y.append(label)
        return np.array(X), np.array(y)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


def get_loader(word2ix):
    train_set = TextDataset(TRAIN_SET_PATH, word2ix, SEQ_LEN, is_train=True)
    validation_set = TextDataset(VALIDATION_SET_PATH, word2ix, SEQ_LEN)
    test_set = TextDataset(TEST_SET_PATH, word2ix, SEQ_LEN)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    return train_loader, validation_loader, test_loader


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=50,
        hidden_dim=32,
        n_layers=2,
        bidirectional=True,
        dropout=0,
        cell="LSTM",
        embedding_matrix=None,
    ):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = 2 if bidirectional else 1

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(embedding_matrix),
                freeze=False,
                padding_idx=0,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        assert cell in ["LSTM", "GRU"]
        if cell == "LSTM":
            cell = nn.LSTM
        elif cell == "GRU":
            cell = nn.GRU

        self.rnn = cell(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * self.bidirectional, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        self.init_weights()
    
    def init_weights(self):
        train_layer = [self.fc, self.rnn]
        for layer in train_layer:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)

    def forward(self, x):
        x = self.embedding(x.long())
        if self.rnn.__class__.__name__ == "LSTM":
            output, (hidden, cell) = self.rnn(x)
        elif self.rnn.__class__.__name__ == "GRU":
            output, hidden = self.rnn(x)
        if self.bidirectional == 2:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :].squeeze()
        x = hidden
        x = self.fc(x)
        return x


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=50,
        dropout=0,
        embedding_matrix=None,
    ):
        super(TextCNN, self).__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(embedding_matrix),
                freeze=False,
                padding_idx=0,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            
        self.cnn = nn.Sequential(
            nn.Conv1d(embedding_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(1920, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        self.init_weights()
    
    def init_weights(self):
        train_layer = [self.fc, self.cnn]
        for layer in train_layer:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)

    def forward(self, x):
        x = self.embedding(x.long())
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def count_correct(output, label):
    output = output.argmax(dim=1)
    TP = ((output == 1) & (label == 1)).sum().item()
    TN = ((output == 0) & (label == 0)).sum().item()
    FP = ((output == 1) & (label == 0)).sum().item()
    FN = ((output == 0) & (label == 1)).sum().item()
    return TP, TN, FP, FN

def get_metrics(TP, TN, FP, FN):
    acc = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return acc, precision, recall, f1


def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    loss = 0
    TP, TN, FP, FN = 0, 0, 0, 0
    for data, label in data_loader:
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss_ = criterion(output, label)
        loss += loss_.item()
        loss_.backward()

        TP_, TN_, FP_, FN_ = count_correct(output, label)
        TP += TP_
        TN += TN_
        FP += FP_
        FN += FN_
        optimizer.step()
    loss /= len(data_loader.dataset)
    loss *= BATCH_SIZE
    acc, precision, recall, f1 = get_metrics(TP, TN, FP, FN)
    return loss, [acc, precision, recall, f1]


def train(model, train_loader, validation_loader, optimizer, criterion, epochs=EPOCH):
    log("Start training...")

    train_loss_list, train_acc_list = [], []
    validation_loss_list, validation_acc_list = [], []

    best_valiadtion_loss = 1e9
    best_validation_acc = 0
    best_epoch = 0

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(1, epochs + 1):
        log("Epoch: {} / {}".format(epoch, epochs))

        loss, metrics = train_epoch(model, train_loader, optimizer, criterion)
        train_loss_list.append(loss)
        train_acc_list.append(metrics[0])
        log(
            "Training:   loss={:.6f}, acc={:.4f}, precision={:.4f}, recall={:.4f}, f1={:.4f}".format(
                loss, *metrics
            )
        )

        loss, metrics = evaluate(model, validation_loader, criterion)
        validation_loss_list.append(loss)
        validation_acc_list.append(metrics[0])
        log(
            "Validation: loss={:.6f}, acc={:.4f}, precision={:.4f}, recall={:.4f}, f1={:.4f}".format(
                loss, *metrics
            )
        )

        if loss < best_valiadtion_loss:
            best_valiadtion_loss = loss
            best_validation_acc = metrics[0]
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_PATH)

            log(
                "Best epoch: {}, best validation loss: {:.6f}, best validation acc: {:.4f}".format(
                    best_epoch, best_valiadtion_loss, best_validation_acc
                )
            )
        scheduler.step()

    log("Training finished.")
    plot_figure(
        train_loss_list, train_acc_list, validation_loss_list, validation_acc_list
    )


def evaluate(model, data_loader, criterion):
    model.eval()
    loss = 0
    TP, TN, FP, FN = 0, 0, 0, 0
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss_ = criterion(output, label)
            loss += loss_.item()
            TP_, TN_, FP_, FN_ = count_correct(output, label)
            TP += TP_
            TN += TN_
            FP += FP_
            FN += FN_
    loss /= len(data_loader.dataset)
    loss *= BATCH_SIZE
    acc, precision, recall, f1 = get_metrics(TP, TN, FP, FN)
    log('TP: {}, TN: {}, FP: {}, FN: {}'.format(TP, TN, FP, FN))
    return loss, [acc, precision, recall, f1]


def plot_figure(
    train_loss_list, train_acc_list, validation_loss_list, validation_acc_list
):
    plt.figure(figsize=(20, 10))
    plt.title("Model: {}".format(MODEL_TYPE))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label="train loss")
    plt.plot(validation_loss_list, label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label="train accuracy")
    plt.plot(validation_acc_list, label="validation accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.savefig(FIGURE_PATH)


def main():
    word2ix = build_vocab()
    embedding_matrix = build_embedding_matrix(word2ix)
    train_loader, validation_loader, test_loader = get_loader(word2ix)

    if MODEL_TYPE in ["LSTM", "GRU"]:
        model = RNN(
            vocab_size=len(word2ix),
            embedding_dim=embedding_matrix.shape[1],
            hidden_dim=HIDDEN_SIZE,
            n_layers=1,
            bidirectional=True,
            dropout=DROPOUT,
            cell=MODEL_TYPE,
            embedding_matrix=embedding_matrix,
        )
    elif MODEL_TYPE == "CNN":
        model = TextCNN(
            vocab_size=len(word2ix),
            embedding_dim=embedding_matrix.shape[1],
            dropout=DROPOUT,
            embedding_matrix=embedding_matrix,
        )

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    if not IS_TEST:
        log("==========================================")
        log("Train shape: {}".format(train_loader.dataset.data.shape))
        log("Validation shape: {}".format(validation_loader.dataset.data.shape))
        log("Test shape: {}".format(test_loader.dataset.data.shape))
        log("Batch size: {}".format(BATCH_SIZE))
        log("Epoch: {}".format(EPOCH))
        log("Learning rate: {}".format(LEARNING_RATE))
        log("Model type: {}".format(MODEL_TYPE))
        log("Embedding matrix shape: {}".format(embedding_matrix.shape))
        log("Vocab size: {}".format(len(word2ix)))
        log("Sequnce length: {}".format(SEQ_LEN))
        log("==========================================")
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train(model, train_loader, validation_loader, optimizer, criterion)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    loss, metrics = evaluate(model, test_loader, criterion)
    log(
        "Test: loss={:.6f}, acc={:.4f}, precision={:.4f}, recall={:.4f}, f1={:.4f}".format(
            loss, *metrics
        )
    )


if __name__ == "__main__":
    main()
