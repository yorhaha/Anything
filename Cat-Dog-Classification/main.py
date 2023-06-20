import torch
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torchsummary import summary
import time
import os

args = ArgumentParser()
args.add_argument('--batch_size', type=int, default=128)
args.add_argument('--epoch', type=int, default=50)
args.add_argument('--learning_rate', type=float, default=1e-3)
args.add_argument('--data_path', type=str, default='./data/train')
args.add_argument('--model_path', type=str, default='./model.pth')
args.add_argument('--log_path', type=str, default='./model.log')
args.add_argument('--figure_path', type=str, default='./figure.png')
args.add_argument('--dropout', type=float, default=0.25)
args.add_argument('--test', type=bool, default=False)
args = args.parse_args()

BATCH_SIZE = args.batch_size
EPOCH = args.epoch
MODEL_PATH = args.model_path
LOG_PATH = args.log_path
FIGURE_PATH = args.figure_path
IS_TEST = args.test
LEARNING_RATE = args.learning_rate
DROP_OUT = args.dropout

logger = open(LOG_PATH, 'a')

def log(msg: str):
    print(msg)
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    logger.write('[{}]{}\n'.format(time_str, msg))
    logger.flush()

log('=========== Initialization ===========')
log('Batch size: {}'.format(BATCH_SIZE))
log('Learning rate: {}'.format(LEARNING_RATE))
log('Dropout: {}'.format(DROP_OUT))
log('Epoch: {}'.format(EPOCH))

CAT, DOG = 0, 1

class MyDataset(Dataset):
    def __init__(self, data_path, label, transform=None):
        self.data_path = data_path
        self.files = os.listdir(self.data_path)
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.label = label
    
    def __getitem__(self, index):
        file = self.files[index]
        img = plt.imread(os.path.join(self.data_path, file))
        img = self.transform(img)
        return img, self.label
    
    def __len__(self):
        return len(self.files)

def get_dataloader(data_path, batch_size=BATCH_SIZE):
    cat_dataset = MyDataset(os.path.join(data_path, 'cats'), CAT)
    dog_dataset = MyDataset(os.path.join(data_path, 'dogs'), DOG)
    dataset = torch.utils.data.ConcatDataset([cat_dataset, dog_dataset])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(DROP_OUT),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(DROP_OUT),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(DROP_OUT),
        )
        self.out = nn.Sequential(
            nn.Linear(64 * 3 * 3, 32),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Linear(32, 2),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
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
        log('Epoch: {}/{}'.format(epoch, epochs))
        loss, accuracy = train_epoch(model, train_loader, optimizer, criterion, device)
        train_loss_list.append(loss)
        train_accuracy_list.append(accuracy)

        # for test
        loss, accuracy = evaluate(model, test_loader, criterion, device, True)
        test_loss_list.append(loss)
        test_accuracy_list.append(accuracy)

        if loss < min_test_loss:
            min_test_loss = loss
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_PATH)
        log('Best epoch: {}'.format(best_epoch))
    
    log('Training finished')
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='train loss')
    plt.plot(test_loss_list, label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy_list, label='train accuracy')
    plt.plot(test_accuracy_list, label='validation accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.savefig(FIGURE_PATH)

def evaluate(model, data_loader, criterion, device, validation=False):
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            correct += output.argmax(dim=1, keepdim=True).eq(target.view_as(output.argmax(dim=1, keepdim=True))).sum().item()
    loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    if validation:
        log('Validation: loss: {:.6f}, accuracy: {:.4f}'.format(loss, accuracy))
    else:
        log('Test: loss: {:.6f}, accuracy: {:.4f}'.format(loss, accuracy))

    return loss, accuracy

def show_error_images(model, data_loader, device):
    model.eval()
    num_wrong_dog, num_wrong_cat = 0, 0
    wrong_dog_list, wrong_cat_list = [], []
    ANIMAL = ['Cat', 'Dog']
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.argmax(dim=1, keepdim=True)
            for i, (o, t) in enumerate(zip(output, target)):
                if o != t:
                    if t.item() == DOG:
                        num_wrong_dog += 1
                        if len(wrong_dog_list) < 10:
                            wrong_dog_list.append((data[i].cpu(), o.item(), t.item()))
                    else:
                        num_wrong_cat += 1
                        if len(wrong_cat_list) < 10:
                            wrong_cat_list.append((data[i].cpu(), o.item(), t.item()))
    print(f'wrong dog: {num_wrong_dog}, wrong cat: {num_wrong_cat}')
    wrong_list = wrong_dog_list + wrong_cat_list
    plt.figure(figsize=(10, 8))
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(wrong_list[i][0].squeeze(0).permute(1, 2, 0))
        plt.axis('off')
    plt.savefig('error_images.png')

def main():
    train_loader = get_dataloader('data/train')
    validation_loader = get_dataloader('data/validation')
    test_loader = get_dataloader('data/test')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))
    model = CNN().to(device)
    summary(model, (3, 224, 224))

    criterion = torch.nn.CrossEntropyLoss()

    if not IS_TEST:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train(model, train_loader, validation_loader, optimizer, criterion, device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    evaluate(model, test_loader, criterion, device)
    # show_error_images(model, test_loader, device)

if __name__ == '__main__':
    main()
