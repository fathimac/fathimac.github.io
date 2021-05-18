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

        for i, (inputs, labels) in enumerate(dataloaders['validation']):

            inputs = inputs.to(device)

            labels = labels.to(device)



            outputs = model(inputs)

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


data_transforms = {

    'train': transforms.Compose([

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'validation': transforms.Compose([

        transforms.Resize(256),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'test': transforms.Compose([

        transforms.Resize(256),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),
}

data_dir = '/data/add_disk0/fcherat/full_images/'
PATH='pyresmodel/pyresmodel6.pth'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                  data_transforms[x])
                  for x in ['train', 'validation','test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
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
            num_steps = dataset_sizes[phase]//32+1
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
                step_loss = loss.item() * inputs.size(0) / 32
                step_acc = torch.sum(preds == labels.data) / 32
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


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
new_in_channels = 4
model = models.resnet50(pretrained=True)
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

#model_ft = torch.load("pyresmodel/pyresmodel6.pth")
#visualize_model(model_ft)
