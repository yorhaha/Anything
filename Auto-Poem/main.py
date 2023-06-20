import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import time
import numpy as np


def seed_evetything(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_evetything(3847)

args = ArgumentParser()
args.add_argument("--batch_size", type=int, default=256)
args.add_argument("--epoch", type=int, default=50)
args.add_argument("--data_path", type=str, default="./tang.npz")
args.add_argument("--model_path", type=str, default="./model.pth")
args.add_argument("--log_path", type=str, default="./model.log")
args.add_argument("--figure_path", type=str, default="./figure.png")
args.add_argument("--test", type=bool, default=False)
args.add_argument("--learning_rate", type=float, default=1e-3)
args.add_argument(
    "--model_type",
    type=str,
    default="LSTM",
    choices=["LSTM", "GRU"],
)
args = args.parse_args()

BATCH_SIZE = args.batch_size
EPOCH = args.epoch
DATA_PATH = args.data_path
MODEL_PATH = args.model_path
LOG_PATH = args.log_path
FIGURE_PATH = args.figure_path
IS_TEST = args.test
LEARNING_RATE = args.learning_rate
MODEL_TYPE = args.model_type

logger = open(LOG_PATH, "a")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str):
    print(msg)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    logger.write("{}|{}\n".format(time_str, msg))
    logger.flush()


with np.load(DATA_PATH, allow_pickle=True) as tang:
    data = tang["data"]
    ix2word = tang["ix2word"].item()
    word2ix = tang["word2ix"].item()


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size=9000,
        embedding_dim=64,
        hidden_dim=32,
        n_layers=2,
        bidirectional=False,
        dropout=0,
        cell="LSTM",
    ):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = 2 if bidirectional else 1

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
            nn.Linear(hidden_dim * self.bidirectional, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, vocab_size),
        )

    def init_hidden(self, batch_size=1):
        return (
            torch.zeros(self.n_layers * self.bidirectional, batch_size, self.hidden_dim).to(device),
            torch.zeros(self.n_layers * self.bidirectional, batch_size, self.hidden_dim).to(device),
        )

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.shape[0])
        else:
            assert isinstance(hidden, tuple) and len(hidden) == 2

        x = self.embedding(x.long())
        x, hidden = self.rnn(x, hidden)
        x = self.fc(x)

        x = x.view(-1, x.shape[-1])
        return x, hidden


def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    loss = 0
    for data in tqdm(data_loader):
        source = data[0].to(device)
        target = data[1].to(device)

        optimizer.zero_grad()
        output, hidden = model(source)
        loss_ = criterion(output, target.view(-1))
        loss += loss_.item()
        loss_.backward()
        optimizer.step()
    loss /= len(data_loader.dataset)
    log("Train: loss: {:.9f}".format(loss))
    return loss


def train(model, data_loader, optimizer, criterion, epochs=EPOCH):
    log("Start training...")

    train_loss_list = []
    min_train_loss = 1e9
    best_epoch = 0

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(1, epochs + 1):
        log("Epoch: {} / {}".format(epoch, epochs))
        loss = train_epoch(model, data_loader, optimizer, criterion)
        train_loss_list.append(loss)
        scheduler.step()

        if loss < min_train_loss:
            min_train_loss = loss
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_PATH)
        log("Best epoch: {}".format(best_epoch))

        generate(model, ix2word, word2ix)
    log("Training finished.")

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1)
    plt.plot(train_loss_list, label="train loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss")

    plt.savefig(FIGURE_PATH)


def generate(model, start_seq="", head_words=[], seq_len=5, total_seq=8, top_k=5):
    if head_words:
        total_seq = len(head_words)
    
    model.eval()
    with torch.no_grad():
        x = torch.Tensor([word2ix["<START>"]]).view(1, 1).to(device)
        hidden = model.init_hidden(1)
        poem = ""
        for i in range(125):
            output, hidden = model(x, hidden)

            n_seq = (i + 1) // (seq_len + 1)
            is_start = (i + 1) % (seq_len + 1) == 1
            is_end = (i + 1) % (seq_len + 1) == 0
            is_whole_start = is_start and n_seq % 2 == 0
            is_whole_end = is_end and n_seq % 2 == 0
            is_half_start = is_start and n_seq % 2 != 0
            is_half_end = is_end and n_seq % 2 != 0

            if i < len(start_seq):
                # Use start_seq if specified
                idx = word2ix[start_seq[i]]
            
            elif is_end:
                # Use punctuation if is end of a sentence
                if is_half_end:
                    idx = word2ix["，"]
                elif is_whole_end:
                    idx = word2ix["。"]
                else:
                    assert False
            
            elif is_whole_start and n_seq == total_seq:
                # Use <EOP> if is the end of the last sentence
                idx = word2ix["<EOP>"]
            
            elif is_start and head_words:
                # Use head_words if specified
                idx = word2ix[head_words[n_seq]]
            
            else:
                # Use the character with the highest probability
                output = output.view(-1)
                output = torch.softmax(output, dim=0)

                # Remove characters that are not allowed to appear
                output[word2ix["</s>"]] = 0
                output[word2ix["<START>"]] = 0
                output[word2ix["<EOP>"]] = 0
                output[word2ix["，"]] = 0
                output[word2ix["。"]] = 0
                
                prob, idx = torch.max(output, dim=0)
                for i in range(torch.randint(0, top_k, (1,)).item()):
                    output[idx.item()] = 0
                    prob, idx = torch.max(output, dim=0)
                idx = idx.item()

            if idx == word2ix["<EOP>"]:
                break
            
            poem += ix2word[idx]
            x = torch.Tensor([idx]).view(1, 1).to(device)
        log(poem.replace('，', '，\n').replace('。', '。\n'))


class PoemDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super(PoemDataset, self).__init__()
        self.seq_len = 48
        self.data = self.proprecess(data)

    def proprecess(self, data):
        data = data.reshape(-1).tolist()
        data = [d for d in data if d != 8292]
        data = np.array(data)
        return data

    def __getitem__(self, index):
        source = self.data[index * self.seq_len : (index + 1) * self.seq_len]
        target = self.data[index * self.seq_len + 1 : (index + 1) * self.seq_len + 1]
        source = torch.from_numpy(source).long()
        target = torch.from_numpy(target).long()
        return source, target

    def __len__(self):
        return self.data.shape[0] // self.seq_len


def get_loader():
    dataset = PoemDataset(data)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    return data_loader


def test_poem(model):
    generate(model, seq_len=5, top_k=1)
    generate(model, seq_len=5, top_k=3)
    generate(model, seq_len=5, top_k=5)
    generate(model, seq_len=5, top_k=10)
    print("==========================================")
    generate(model, seq_len=7, top_k=1)
    generate(model, seq_len=7, top_k=3)
    generate(model, seq_len=7, top_k=5)
    generate(model, seq_len=7, top_k=10)
    print("==========================================")
    generate(model, start_seq="好", seq_len=5, top_k=1)
    generate(model, start_seq="好", seq_len=5, top_k=3)
    generate(model, start_seq="好", seq_len=5, top_k=5)
    generate(model, start_seq="好", seq_len=5, top_k=10)
    print("==========================================")
    generate(model, start_seq="好雨知", seq_len=5, top_k=1)
    generate(model, start_seq="好雨知", seq_len=5, top_k=3)
    generate(model, start_seq="好雨知", seq_len=5, top_k=5)
    generate(model, start_seq="好雨知", seq_len=5, top_k=10)
    print("==========================================")
    generate(model, start_seq="好雨知时节", seq_len=5, top_k=1)
    generate(model, start_seq="好雨知时节", seq_len=5, top_k=3)
    generate(model, start_seq="好雨知时节", seq_len=5, top_k=5)
    generate(model, start_seq="好雨知时节", seq_len=5, top_k=10)
    print("==========================================")
    generate(model, start_seq="雪", seq_len=5, top_k=1)
    generate(model, start_seq="雪", seq_len=5, top_k=3)
    generate(model, start_seq="雪", seq_len=5, top_k=5)
    generate(model, start_seq="雪", seq_len=5, top_k=10)
    print("==========================================")
    generate(model, start_seq="白", seq_len=5, top_k=1)
    generate(model, start_seq="白", seq_len=5, top_k=3)
    generate(model, start_seq="白", seq_len=5, top_k=5)
    generate(model, start_seq="白", seq_len=5, top_k=10)
    print("==========================================")
    generate(model, head_words="深度学习", seq_len=5, top_k=1)
    generate(model, head_words="深度学习", seq_len=5, top_k=3)
    generate(model, head_words="深度学习", seq_len=5, top_k=5)
    generate(model, head_words="深度学习", seq_len=5, top_k=10)
    print("==========================================")


def main():
    data_loader = get_loader()

    model = RNN(
        vocab_size=len(ix2word),
        embedding_dim=128,
        hidden_dim=512,
        n_layers=3,
        bidirectional=False,
        dropout=0.1,
        cell=MODEL_TYPE,
    )

    model = model.to(device)

    if not IS_TEST:
        log("==========================================")
        log("Data shape: {}".format(data_loader.dataset.data.shape))
        log("Batch size: {}".format(BATCH_SIZE))
        log("Epoch: {}".format(EPOCH))
        log("Learning rate: {}".format(LEARNING_RATE))
        log("==========================================")
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.CrossEntropyLoss()
        train(model, data_loader, optimizer, criterion)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    test_poem(model)


if __name__ == "__main__":
    main()
