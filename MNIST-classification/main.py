import torch
import torchvision
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary
from argparse import ArgumentParser
import time

def seed_evetything(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_evetything(3847)

args = ArgumentParser()
args.add_argument('--batch_size', type=int, default=128)
args.add_argument('--epoch', type=int, default=50)
args.add_argument('--model_path', type=str, default='./model.pth')
args.add_argument('--log_path', type=str, default='./model.log')
args.add_argument('--figure_path', type=str, default='./figure.png')
args.add_argument('--test', type=bool, default=False)
args.add_argument('--learning_rate', type=float, default=1e-3)
args = args.parse_args()

BATCH_SIZE = args.batch_size
EPOCH = args.epoch
MODEL_PATH = args.model_path
LOG_PATH = args.log_path
FIGURE_PATH = args.figure_path
IS_TEST = args.test
LEARNING_RATE = args.learning_rate

logger = open(LOG_PATH, 'a')

def log(msg: str):
    print(msg)
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    logger.write('{}|{}\n'.format(time_str, msg))

log('==========================================')
log('Batch size: {}'.format(BATCH_SIZE))
log('Epoch: {}'.format(EPOCH))
log('Learning rate: {}'.format(LEARNING_RATE))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cov1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.cov1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    loss, correct = 0, 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_ = criterion(output, target)
        loss += loss_.item()
        correct += output.argmax(dim=1, keepdim=True).eq(target.view_as(output.argmax(dim=1, keepdim=True))).sum().item()
        loss_.backward()
        optimizer.step()
    loss /= len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    log('Train: loss: {:.6f}, accuracy: {:.4f}'.format(loss, accuracy))
    return loss, accuracy

def train(model, train_loader, test_loader, optimizer, criterion, device, epochs=EPOCH):
    log('Start training...')

    train_loss_list, train_accuracy_list = [], []
    test_loss_list, test_accuracy_list = [], []
    min_test_loss = 1e9
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # for train
        log('Epoch: {}'.format(epoch))
        loss, accuracy = train_epoch(model, train_loader, optimizer, criterion, device)
        train_loss_list.append(loss)
        train_accuracy_list.append(accuracy)

        # for test
        loss, accuracy = evaluate(model, test_loader, criterion, device)
        test_loss_list.append(loss)
        test_accuracy_list.append(accuracy)

        if loss < min_test_loss:
            min_test_loss = loss
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_PATH)
        log('Best epoch: {}'.format(best_epoch))
    
    log('Training finished.')
    
    # set plot size
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='train loss')
    plt.plot(test_loss_list, label='test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy_list, label='train accuracy')
    plt.plot(test_accuracy_list, label='test accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.savefig(FIGURE_PATH)

def evaluate(model, data_loader, criterion, device):
    model.eval()
    loss, correct = 0, 0
    wrong_list = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            output = output.argmax(dim=1, keepdim=True)
            for i, (o, t) in enumerate(zip(output, target)):
                if o != t:
                    wrong_list.append((data[i], o, t))
            correct += output.eq(target.view_as(output)).sum().item()
    loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    log(' Test: loss: {:.6f}, accuracy: {:.4f}'.format(loss, accuracy))

    return loss, accuracy

def get_loader():
    train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader

def show_error_images(model, data_loader, criterion, device):
    model.eval()
    wrong_list = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.argmax(dim=1, keepdim=True)
            for i, (o, t) in enumerate(zip(output, target)):
                if o != t:
                    wrong_list.append((data[i], o.item(), t.item()))
    print('Number of wrong images: {}'.format(len(wrong_list)))
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(wrong_list[i][0].reshape(28, 28), cmap='gray')
        plt.title('pred: {}, label: {}'.format(wrong_list[i][1], wrong_list[i][2]))
        plt.axis('off')
    plt.savefig('error_images.png')

def main():
    train_loader, test_loader = get_loader()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    summary(model, (1, 28, 28))

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    if not IS_TEST:
        train(model, train_loader, test_loader, optimizer, criterion, device)
    else:
        model.load_state_dict(torch.load(MODEL_PATH))
        evaluate(model, test_loader, criterion, device)
        show_error_images(model, test_loader, criterion, device)

if __name__ == '__main__':
    main()
