import matplotlib.pyplot as plt
import numpy as np
import os

import torch
from torchvision import datasets, transforms
#from data.gen_language_data import load_language_data
#from data.gen_multnist import load_multnist_data
#from data.gen_dolphin_data import load_dolphin_data

# === set seeds ===
'''
random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
np.random.seed(7)
torch.backends.cudnn.deterministic=True
'''


# === CUTOUT ===========================================================================================================
# adapted from github.com/hysts/pytorch_cutout
class Cutout:
    def __init__(self, mask_size, p, mask_color=(0, 0, 0)):
        self.mask_size = mask_size
        self.p = p
        self.mask_color = mask_color
        self.mask_size_half = mask_size // 2
        self.offset = 1 if mask_size % 2 == 0 else 0

    def __call__(self, image):
        image = np.asarray(image).copy()

        if np.random.random() > self.p:
            return image

        h, w = image.shape[:2]

        cxmin, cxmax = 0, w + self.offset
        cymin, cymax = 0, h + self.offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - self.mask_size_half
        ymin = cy - self.mask_size_half
        xmax = xmin + self.mask_size
        ymax = ymin + self.mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = self.mask_color[0]*255.
        return image


# === DATA HELPERS =====================================================================================================
def load_data(batch_size, dataset, metadata=None):
    data_path = os.getcwd() + "/data/"
    download = dataset not in os.listdir(data_path)

    # dataset configuration
    if dataset == 'MultNIST':
        return load_multnist_data(batch_size)
    elif dataset == 'Language':
        return load_language_data()
    elif dataset == 'Dolphin':
        train_data, test_data, MEAN, STD = load_dolphin_data()
        train_xs, train_ys = train_data
        test_xs, test_idx, test_ys = test_data
        
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            Cutout(mask_size=16, p=.5, mask_color=MEAN),
            transforms.ToTensor(),
            transforms.Normalize(MEAN/255, STD/255)])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN/255, STD/255)])
        train_xs = [train_transform(x) for x in train_xs]
        test_xs = [test_transform(x) for x in test_xs]
        
        train_data = list(zip(train_xs, train_ys))
        test_data = list(zip(test_xs, test_idx, test_ys))
        del train_xs, train_ys, test_xs, test_ys, test_idx
        
    elif dataset == 'CIFAR10':
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD  = [0.24703233, 0.24348505, 0.26158768]

        train_data = datasets.CIFAR10(data_path+dataset,
                                      train=True,
                                      download=download,
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          Cutout(mask_size=16, p=.5, mask_color=MEAN),
                                          transforms.ToTensor(),
                                          transforms.Normalize(MEAN, STD)]
                                      ))
        test_data = datasets.CIFAR10(data_path+dataset,
                                     train=False,
                                     download=download,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize(MEAN, STD)]
                                     ))
        valid_size = 10*batch_size
        train_subset  = torch.utils.data.Subset(train_data, range(0, len(train_data)-valid_size))
        valid_data = torch.utils.data.Subset(train_data, range(len(train_data)-valid_size, len(train_data)))
        train_data = train_subset
    elif dataset == "MNIST":
        train_data = datasets.MNIST(data_path+dataset,
                                    train=True, 
                                    download=download,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))]
                                    ))
        test_data = datasets.MNIST(data_path+dataset, 
                                   train=False,
                                   download=download,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))]
                                   ))
        
    elif dataset == 'ImageNet':
        MEAN = [0.485, 0.456, 0.406]
        STD  = [0.229, 0.224, 0.225]

        train_data = datasets.ImageNet(data_path+dataset,
                                      split='train',
                                      download=False,
                                      transform=transforms.Compose([
                                          transforms.RandomResizedCrop(112),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(
                                              brightness=0.4,
                                              contrast=0.4,
                                              saturation=0.4,
                                              hue=0.2),
                                          transforms.ToTensor(),
                                          transforms.Normalize(MEAN, STD)]
                                      ))
        test_data = datasets.ImageNet(data_path+dataset,
                                     split='val',
                                     download=False,
                                     transform=transforms.Compose([
                                         transforms.Resize(128),
                                         transforms.CenterCrop(112),
                                         transforms.ToTensor(),
                                         transforms.Normalize(MEAN, STD)]
                                     ))
    else:
        raise ValueError("No matching dataset configuration: {}".format(dataset))

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                              batch_size=batch_size,
                                              shuffle=False)
    del train_data, test_data, valid_data
        
    # get/load dataset metadata
    for val in train_loader:
        if len(val) == 2:
            img, target = val
        elif len(val) == 3:
            img, info, target = val
        data_shape = img.shape
        break

    return [train_loader, test_loader, valid_loader], data_shape


# === AUGMENTATION PREVIEW =============================================================================================
def vis_im(im):
    plt.figure(figsize=(1,1))
    t_image = im.transpose(0,2).transpose(0,1)
    t_image = (t_image-torch.min(t_image))/(torch.max(t_image)-torch.min(t_image))
    ax = plt.gca()
    ax.axis('off')
    plt.imshow(t_image)

def visualize_loader(loader):
    plt.figure()
    for data, target in loader:
        for i, image in enumerate(data[0:9]):
            print(torch.min(image),torch.max(image),torch.mean(image))
            ax = plt.subplot(3,3,i+1)
            t_image = image.transpose(0,2).transpose(0,1)
            t_image = (t_image-torch.min(t_image))/(torch.max(t_image)-torch.min(t_image))
            print(torch.min(t_image),torch.max(t_image),torch.mean(t_image))
            ax.imshow(t_image)
            ax.axis('off')
        break

    plt.show()


def get_sample_batch(loader):
    for data, target in loader:
        return data

