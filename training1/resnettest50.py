import torch.nn as nn
import torch
from torchvision import models, datasets, transforms, utils
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import cv2
from PIL import Image


global y_pred_list
global y_test
global incorrect


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp1 = 'C:\\Users\\fathi\\Documents\\Fall_20\\theseis\\test\\test\\' + inp + '.jpg'
    inp = cv2.imread(inp1)
    #inp = inp.reshape((256, 256, 4))
    #inp = inp[:, :, :3]
    #img = Image.fromarray(inp, 'RGB')
    #img.save('abc.jpg')
    if inp is not None:
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # inp = std * inp + mean
        # inp = np.clip(inp, 0, 1)

        plt.imshow(inp)

    #if title is not None:
     #   plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, num_images=64):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    k = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs1 = inputs[0].to(device)
            labels = labels.to(device)

            outputs = model(inputs1.float())
            _, preds = torch.max(outputs, 1)
            for j in range(inputs1.size()[0]):
                if class_names[preds[j]] != class_names[labels[j]] and labels[j] == 0:
                    images_so_far += 1
                    ax = plt.subplot(num_images // 8, 8, images_so_far)
                    ax.axis('off')
                    #ax.set_title('{},{}'.format(class_names[preds[j]], labels[j]))
                    imshow(inputs[1][j])
                    if images_so_far == num_images:
                        plt.savefig('1NotDetected-'+str(k)+'.png')
                        k += 1
                        images_so_far = 0
                        plt.close(fig)
                        fig = plt.figure()
        model.train(mode=was_training)


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = torch.from_numpy(img_path)
        label = self.img_labels.iloc[idx, 1]
        sample = {"image": image, "label": label}
        return sample


def model_eval(model):
    global y_pred_list
    global y_test
    global incorrect
    model.eval()
    running_loss = 0.0

    running_corrects = 0
    phase = 'test'
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Iterate over data.

    for inputs, labels in dataloaders[phase]:
        inputs1 = inputs.to(device)

        labels = labels.to(device)

        # zero the parameter gradients

        optimizer.zero_grad()

        # forward

        # track history if only in train

        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs1.float())

            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            step_loss = loss / 64

            step_acc = torch.sum(preds == labels.data) / 64
            #print('{} sLoss: {:.4f} sAcc: {:.4f}'.format(
            #    phase, step_loss, step_acc))
            y_test.extend(labels.cpu().numpy())
            y_pred_list.extend(preds.cpu().numpy())
            
        # statistics

        running_loss += loss.item() * inputs.size(0)

        running_corrects += torch.sum(preds == labels.data)

    loss = running_loss / dataset_sizes[phase]

    acc = running_corrects.double() / dataset_sizes[phase]

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(

        phase, loss, acc))
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_test = [a.squeeze().tolist() for a in y_test]
    return acc


def normalize(tensor, mean, std, inplace=False):

    """Normalize a tensor image with mean and standard deviation.
    .. note::

        This transform acts out of place by default, i.e., it does not mutates the input tensor.
    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
      Tensor: Normalized Tensor image.

    """
    if not torch.is_tensor(tensor):
        raise TypeError('tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndimension() != 3:
        raise ValueError('Expected tensor to be a tensor image of size (C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]
    tensor.sub_(mean).div_(std)
    return tensor


class Normalize(object):

    """Normalize a tensor image with mean and standard deviation.

    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``

    channels, this transform will normalize each channel of the input

    ``torch.*Tensor`` i.e.,

    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::

        This transform acts out of place, i.e., it does not mutate the input tensor.
    Args:

        mean (sequence): Sequence of means for each channel.

        std (sequence): Sequence of standard deviations for each channel.

        inplace(bool,optional): Bool to make this operation in-place.

    """


    def __init__(self, mean, std, inplace=False):

        self.mean = mean
        self.std = std
        self.inplace = inplace


    def __call__(self, tensor):

        """

        Args:

            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:

            Tensor: Normalized Tensor image.

        """
        return normalize(tensor, self.mean, self.std, self.inplace)


class ToTensor(object):

    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        image = sample
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)



nm = transforms.Normalize(mean=[0.485, 0.456, 0.406],

                             std=[0.229, 0.224, 0.225])

to_tens = ToTensor()

to_te = transforms.ToTensor()
def npy_loader(path):
    img = (cv2.imread(path)).astype(float)
    path1 = (path[-16:]).rstrip('.jpg')
    np_array_path = '/nfs/bignet/add_disk0/fcherat/numpyarray/' + path1 + '_densitymap.npy'
    heat_map = (np.load(np_array_path)).astype(float)
    if len(heat_map.shape) == 2:
        heat_map = np.reshape(heat_map, (256, 256, -1)) 
        #std = heat_map[heat_map != 0].std()
        #mean = heat_map[heat_map != 0].mean()
        #nm = transforms.Normalize(mean=[mean, mean, mean],
        #                  std=[std, std, std])
    #else:
    nm = transforms.Normalize(mean=[0.485, 0.456, 0.406],

                             std=[0.229, 0.224, 0.225])
    heat_map = to_tens(heat_map)
    img  = to_te(img)
    sample = nm(img)
    res =  torch.cat((sample, heat_map), dim=0)
    return res


data_dir = '/nfs/bignet/add_disk0/fcherat/cropped-images'


data_transforms = {

    'train': transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'validation': transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'test': transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

}

image_datasets = {x: datasets.DatasetFolder(os.path.join(data_dir, x),
                  loader=npy_loader,
                  extensions='.jpg')
                  for x in ['train','validation','test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                             shuffle=True) for x in ['train','validation','test']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train','validation','test']}

class_names = image_datasets['train'].classes

PATH = "pyresmodel10.pth"


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            step = 0;
            num_steps = dataset_sizes[phase]//64+1
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs.float())
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                #print('Step {}/{}'.format(step, num_steps - 1))
                #step_loss = loss.item() * inputs.size(0) / 64
                #step_acc = torch.sum(preds == labels.data) / 64
                step += 1
                #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                #    phase, step_loss, step_acc))

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #    phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, PATH)
                best_loss = epoch_loss

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Acc: {:4f}'.format(best_loss))


    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#new_in_channels = 4
#model = models.resnet50(pretrained=True)

#layer = model.conv1

# Creating new Conv2d layer
#new_layer = nn.Conv2d(in_channels=new_in_channels,
#                      out_channels=layer.out_channels,
#                      kernel_size=layer.kernel_size,
#                      stride=layer.stride,
#                      padding=layer.padding,
#                      bias=layer.bias)
#

#copy_weights = 0  # Here will initialize the weights from new channel with the red channel weights

# Copying the weights from the old to the new layer
#new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()

# Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
#for i in range(new_in_channels - layer.in_channels):
#    channel = layer.in_channels + i
#    new_layer.weight[:, channel:channel + 1, :, :] = layer.weight[:, copy_weights:copy_weights + 1, ::].clone()
#new_layer.weight = nn.Parameter(new_layer.weight)

#model.conv1 = new_layer

#num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
#model.fc = nn.Linear(num_ftrs, 2)

#model_ft = model.to(device)

#criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#model_ft = torch.load(PATH)

#model_ft = model.to(device)

best_epoch = 0
best_val = 0
for epoch in range(25):
    paths = 'allmodeldam' + str(epoch) + '.pth'
    model = torch.load(paths)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    y_pred_list=[]
    y_test=[]
    acc = model_eval(model)
    if acc>best_val:
        best_val = acc
        best_epoch = epoch
    print(confusion_matrix(y_test, y_pred_list))
    print(classification_report(y_test, y_pred_list))
  
print(best_epoch)
print(best_val)
