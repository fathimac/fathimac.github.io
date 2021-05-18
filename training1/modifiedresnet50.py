import torch.nn as nn
import torch
from torchvision import models,datasets, transforms
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs.float())
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
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


def npy_loader(path):
    sample = torch.from_numpy(np.reshape(np.load(path), (4,256,256)))
    return sample


data_dir = '/nfs/bignet/add_disk0/fcherat/NumpyFiles1'
PATH='pyresmodel/pyresmodel4.pth'
image_datasets = {x: datasets.DatasetFolder(os.path.join(data_dir, x),
                  loader=npy_loader,
                  extensions='.npy')
                  for x in ['train', 'validation','test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                             shuffle=True, num_workers=4) for x in ['train', 'validation','test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation','test']}
class_names = image_datasets['train'].classes


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
                print('Step {}/{}'.format(step, num_steps - 1))
                step_loss = loss.item() * inputs.size(0) / 64
                step_acc = torch.sum(preds == labels.data) / 64
                step += 1
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, step_loss, step_acc))

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            paths = 'allmod'+str(epoch)+'.pth'
            torch.save(model,paths)
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
new_in_channels = 4
model = models.resnet50(pretrained=True)

layer = model.conv1     
    
# Creating new Conv2d layer
new_layer = nn.Conv2d(in_channels=new_in_channels,
                      out_channels=layer.out_channels,
                      kernel_size=layer.kernel_size,
                      stride=layer.stride,
                      padding=layer.padding,
                      bias=layer.bias)
                                                                                                                                                    
copy_weights = 0  # Here will initialize the weights from new channel with the red channel weights

# Copying the weights from the old to the new layer
new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()

# Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
for i in range(new_in_channels - layer.in_channels):
    channel = layer.in_channels + i
    new_layer.weight[:, channel:channel + 1, :, :] = layer.weight[:, copy_weights:copy_weights + 1, ::].clone()
new_layer.weight = nn.Parameter(new_layer.weight)

model.conv1 = new_layer

num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 2)

model_ft = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

#visualize_model(model_ft)
