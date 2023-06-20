import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from train import CNN


def filter_show(model, model_name):
    k = 0
    for param in model.parameters():
        data = param.data.cpu()
        if len(data.shape) == 4:
            fig, axs = plt.subplots(data.shape[0], data.shape[1], figsize=(12, 15))
            if data.shape[1] == 1:
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        axs[i].imshow(data[i, j, :, :], cmap='Greys')  # gray = 0, black > 0, white < 0
                        axs[i].axis('off')
            else:
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        axs[i, j].imshow(data[i, j, :, :], cmap='Greys')  # gray = 0, black > 0, white < 0
                        axs[i, j].axis('off')
            plt.savefig('filter imgs/' + model_name + ' ' + str(k) + '.png')
            k += 1


if __name__ == '__main__':
    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    train = torchvision.datasets.CIFAR100(root='D:\datasets\CIFAR-100',
                                          train=True, transform=train_transform,
                                          download=True)
    test = torchvision.datasets.CIFAR100(root='D:\datasets\CIFAR-100',
                                         train=False, transform=test_transform,
                                         download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=100, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=100, shuffle=False)

    cuda = torch.device('cuda')

    model0, model10, model100 = CNN(), CNN(), CNN()  # init model, trained model
    model0, model10, model100 = model0.cuda(), model10.cuda(), model100.cuda()

    model0.load_state_dict(torch.load("D:/User_Data/Desktop/github/CNN Visualization/CIFAR-100/model/" + str(0) + ".pth"))
    model10.load_state_dict(torch.load("D:/User_Data/Desktop/github/CNN Visualization/CIFAR-100/model/" + str(10) + ".pth"))
    model100.load_state_dict(torch.load("D:/User_Data/Desktop/github/CNN Visualization/CIFAR-100/model/" + str(100) + ".pth"))
    model0.eval()
    model10.eval()
    model100.eval()

    filter_show(model0, 'model0')  # init model
    filter_show(model10, 'model10')  # trained model
    filter_show(model100, 'model100')

"""
weights shape
    torch.Size([32, 1, 3, 3]) = 1st layer filters
    torch.Size([32]) = b
    torch.Size([32, 32, 3, 3]) = 2nd layer filters
    torch.Size([32]) = b
    torch.Size([128, 1568]) = fc layer
    torch.Size([128]) = b
    torch.Size([10, 128]) = fc
    torch.Size([10]) = predict
"""