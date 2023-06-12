import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from train import CNN


if __name__ == '__main__':
    train = torchvision.datasets.MNIST(
        'D:/datasets', train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    )
    test = torchvision.datasets.MNIST(
        'D:/datasets', train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    )

    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=100, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=train, batch_size=100, shuffle=False)

    cuda = torch.device('cuda')

    model0, model10 = CNN(), CNN()
    model0, model10 = model0.cuda(), model10.cuda()

    model0.load_state_dict(torch.load("D:/User_Data/Desktop/github/CNN Visualization/model/" + str(0) + ".pth"))
    model10.load_state_dict(torch.load("D:/User_Data/Desktop/github/CNN Visualization/model/" + str(10) + ".pth"))
    model0.eval()
    model10.eval()

    for param in model10.parameters():
        data = param.data.cpu()
        if len(data.shape)>1:
            fig, axs = plt.subplots(2, 16, figsize=(15, 2))
            for i in range(2):
                for j in range(16):
                    for k in range(len(data[0])):
                        axs[i, j].imshow(data[i * 16 + j, k, :, :], vmin=-0.5, vmax=0.5)
                        axs[i, j].axis('off')
            plt.show()

"""
    torch.Size([32, 1, 3, 3])
    torch.Size([32])
    torch.Size([32, 32, 3, 3])
    torch.Size([32])
    torch.Size([128, 1568])
    torch.Size([128])
    torch.Size([10, 128])
    torch.Size([10])
"""

"""
    correct = 0
    for data, target in test_loader:
        data = data.to(cuda)
        target = target.to(cuda)
        output = model0(data)
        prediction = output.data.max(1)[1]
        correct += prediction.eq(target.data).sum()

    print('Test set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))

    correct = 0
    for data, target in test_loader:
        data = data.to(cuda)
        target = target.to(cuda)
        output = model10(data)
        prediction = output.data.max(1)[1]
        correct += prediction.eq(target.data).sum()

    print('Test set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))
"""