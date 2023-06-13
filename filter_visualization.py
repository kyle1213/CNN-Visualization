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
            fig, axs = plt.subplots(data.shape[0], data.shape[1], figsize=(5, 2))
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
            plt.savefig('filter imgs 5/' + model_name + ' ' + str(k) + '.png')
            k += 1


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

    model0, model10, model100 = CNN(), CNN(), CNN()  # init model, trained model
    model0, model10, model100 = model0.cuda(), model10.cuda(), model100.cuda()

    model0.load_state_dict(torch.load("D:/User_Data/Desktop/github/CNN Visualization/model/" + str(0) + ".pth"))
    model10.load_state_dict(torch.load("D:/User_Data/Desktop/github/CNN Visualization/model/" + str(10) + ".pth"))
    model100.load_state_dict(torch.load("D:/User_Data/Desktop/github/CNN Visualization/model/" + str(100) + ".pth"))
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