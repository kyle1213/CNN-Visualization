import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from train import CNN


def filter_show(model):
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
            plt.show()


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


if __name__ == '__main__':
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    test = torchvision.datasets.CIFAR100(root='D:\datasets\CIFAR-100',
                                         train=False, transform=test_transform,
                                         download=True)

    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=100, shuffle=False)

    cuda = torch.device('cuda')

    data, _ = test[5]
    data = data.to(cuda)
    data.unsqueeze_(0)

    model0, model10, model100 = CNN(), CNN(), CNN()  # init model, trained model
    model0, model10, model100 = model0.cuda(), model10.cuda(), model100.cuda()

    model0.load_state_dict(torch.load("D:/User_Data/Desktop/github/CNN Visualization/CIFAR-100/model/" + str(0) + ".pth"))
    model10.load_state_dict(torch.load("D:/User_Data/Desktop/github/CNN Visualization/CIFAR-100/model/" + str(10) + ".pth"))
    model100.load_state_dict(torch.load("D:/User_Data/Desktop/github/CNN Visualization/CIFAR-100/model/" + str(100) + ".pth"))
    model0.eval()
    model10.eval()
    model100.eval()

    activation = {}

    model0.conv1.register_forward_hook(get_activation('conv1'))
    model0.conv2.register_forward_hook(get_activation('conv2'))
    output0 = model0(data)

    plt.imshow(data.squeeze(0).permute(1, 2, 0).detach().cpu(), cmap='Greys')
    plt.show()

    k=0
    act = activation['conv1'].squeeze()
    fig,ax = plt.subplots(act.size(0)//4, 4, figsize=(12, 15))

    for i in range(act.size(0)//4):
            for j in range(4):
               ax[i,j].imshow(act[k].detach().cpu().numpy(), cmap='Greys')
               k+=1
               plt.savefig('feature map/fm1_0.png')

    k=0
    act = activation['conv2'].squeeze()
    fig, ax = plt.subplots(act.size(0) // 4, 4, figsize=(12, 15))
    for i in range(act.size(0)//4):
            for j in range(4):
               ax[i,j].imshow(act[k].detach().cpu().numpy(), cmap='Greys')
               k+=1
               plt.savefig('feature map/fm2_0.png')

    activation = {}

    model10.conv1.register_forward_hook(get_activation('conv1'))
    model10.conv2.register_forward_hook(get_activation('conv2'))
    output10 = model10(data)

    k = 0
    act = activation['conv1'].squeeze()
    fig, ax = plt.subplots(act.size(0) // 4, 4, figsize=(12, 15))

    for i in range(act.size(0) // 4):
        for j in range(4):
            ax[i, j].imshow(act[k].detach().cpu().numpy(), cmap='Greys')
            k += 1
            plt.savefig('feature map/fm1_10.png')

    k = 0
    act = activation['conv2'].squeeze()
    fig, ax = plt.subplots(act.size(0) // 4, 4, figsize=(12, 15))
    for i in range(act.size(0) // 4):
        for j in range(4):
            ax[i, j].imshow(act[k].detach().cpu().numpy(), cmap='Greys')
            k += 1
            plt.savefig('feature map/fm2_10.png')

    activation = {}

    model100.conv1.register_forward_hook(get_activation('conv1'))
    model100.conv2.register_forward_hook(get_activation('conv2'))
    output100 = model100(data)

    k = 0
    act = activation['conv1'].squeeze()
    fig, ax = plt.subplots(act.size(0) // 4, 4, figsize=(12, 15))

    for i in range(act.size(0) // 4):
        for j in range(4):
            ax[i, j].imshow(act[k].detach().cpu().numpy(), cmap='Greys')
            k += 1
            plt.savefig('feature map/fm1_100.png')

    k = 0
    act = activation['conv2'].squeeze()
    fig, ax = plt.subplots(act.size(0) // 4, 4, figsize=(12, 15))
    for i in range(act.size(0) // 4):
        for j in range(4):
            ax[i, j].imshow(act[k].detach().cpu().numpy(), cmap='Greys')
            k += 1
            plt.savefig('feature map/fm2_100.png')