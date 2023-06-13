import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(1568, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


if __name__ == '__main__':
    train = torchvision.datasets.MNIST('D:/datasets', train=True, download=True,
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize((0.1307,), (0.3081,))]))
    test = torchvision.datasets.MNIST('D:/datasets', train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor(),
                                                                    transforms.Normalize((0.1307,), (0.3081,))]))

    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=100, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=100, shuffle=False)

    cuda = torch.device('cuda')

    model = CNN()
    model = model.cuda()

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=5e-4, momentum=0.9)

    train_cost = 0
    test_cost = 0

    train_correct = 0
    test_correct = 0

    iterations = []
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    torch.save(model.state_dict(), "D:/User_Data/Desktop/github/CNN Visualization/model/" + str(0) + ".pth")

    for epoch in range(1, 101):
        model.train()
        train_correct = 0
        for X, Y in train_loader:
            X = X.to(cuda)
            Y = Y.to(cuda)
            optimizer.zero_grad()
            hypo = model(X)
            train_cost = loss(hypo, Y)
            train_cost.backward()
            optimizer.step()
            prediction = hypo.data.max(1)[1]
            train_correct += prediction.eq(Y.data).sum()

        model.eval()
        test_correct = 0
        for data, target in test_loader:
            data = data.to(cuda)
            target = target.to(cuda)
            output = model(data)
            test_cost = loss(output, target)
            prediction = output.data.max(1)[1]
            test_correct += prediction.eq(target.data).sum()

        print("Epoch : {:>4} / cost : {:>.9}".format(epoch, train_cost))
        iterations.append(epoch)
        train_losses.append(train_cost.tolist())
        test_losses.append(test_cost.tolist())
        train_acc.append((100*train_correct/len(train_loader.dataset)).tolist())
        test_acc.append((100*test_correct/len(test_loader.dataset)).tolist())

        torch.save(model.state_dict(), "D:/User_Data/Desktop/github/CNN Visualization/model/" + str(epoch) + ".pth")

    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(cuda)
        target = target.to(cuda)
        output = model(data)
        prediction = output.data.max(1)[1]
        correct += prediction.eq(target.data).sum()

    print('Test set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))

    plt.subplot(121)
    plt.plot(range(1, len(iterations)+1), train_losses, 'b--')
    plt.plot(range(1, len(iterations)+1), test_losses, 'r--')
    plt.subplot(122)
    plt.plot(range(1, len(iterations)+1), train_acc, 'b-')
    plt.plot(range(1, len(iterations)+1), test_acc, 'r-')
    plt.title('loss and accuracy')
    plt.savefig('train result/result.png')