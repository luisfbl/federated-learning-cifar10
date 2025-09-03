from torch.nn import MaxPool2d, Linear, Conv2d, Module, CrossEntropyLoss, Dropout, BatchNorm2d
from torch.nn.functional import relu
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch import Tensor, no_grad
from torch.utils.data import DataLoader

class CNet(Module):
    def __init__(self):
        super(CNet, self).__init__()
        self.conv1 = Conv2d(3, 32, 5, padding=2)
        self.bn1 = BatchNorm2d(32)
        self.conv2 = Conv2d(32, 64, 5, padding=2)
        self.bn2 = BatchNorm2d(64)
        self.conv3 = Conv2d(64, 128, 3, padding=1)
        self.bn3 = BatchNorm2d(128)

        self.fc1 = Linear(128 * 4 * 4, 256)
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, 10)

        self.pool = MaxPool2d(2, 2)
        self.dropout = Dropout(0.5)

    def forward(self, x: Tensor):
        x = self.pool(relu(self.bn1(self.conv1(x))))
        x = self.pool(relu(self.bn2(self.conv2(x))))
        x = self.pool(relu(self.bn3(self.conv3(x))))

        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(relu(self.fc1(x)))
        x = self.dropout(relu(self.fc2(x)))
        x = self.fc3(x)
        return x

def train(net: CNet, trainloader: DataLoader, epochs: int, verbose=False):
    criterion = CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    net.train()

    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0

        for batch in trainloader:
            images, labels = batch['img'].to('cpu'), batch['label'].to('cpu')
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            epoch_loss += loss.item()

        scheduler.step()

        if verbose:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(trainloader):.4f}, Accuracy: {correct/total:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

def test(net: CNet, testloader: DataLoader):
    criterion = CrossEntropyLoss()
    correct, total, total_loss = 0, 0, 0.0
    net.eval()

    with no_grad():
        for batch in testloader:
            images, labels = batch['img'].to('cpu'), batch['label'].to('cpu')
            outputs = net(images)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            total_loss += loss.item()

    accuracy = correct / total
    avg_loss = total_loss / len(testloader)
    return avg_loss, accuracy
