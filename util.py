import torch
from torchvision import transforms, datasets
import json
import copy
import os


def device_type(gpu=True):
    return torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

def load_categories(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return category_names

def load_datasets(data_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}

    image_datasets = {i: datasets.ImageFolder(os.path.join(data_dir, i), data_transforms[i])
                      for i in ['train', 'valid', 'test']}
    return image_datasets

def train_model(model, criterion, optimizer, scheduler, image_datasets, gpu, epochs=10):
    device = device_type(gpu)
    dataloaders = {i: torch.utils.data.DataLoader(image_datasets[i],
                                                  batch_size=64,
                                                  shuffle=True)
                   for i in ['train', 'valid', 'test']}
    dataset_sizes = {i: len(image_datasets[i]) for i in ['train', 'valid', 'test']}

    network = copy.deepcopy(model.state_dict())
    accuracy = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch +1 ,epochs))

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
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
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_accuracy = running_corrects.double() / dataset_sizes[phase]
    
            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(
                phase, epoch_loss, epoch_accuracy))

            # deep copy the model
            if phase == 'valid' and epoch_accuracy > accuracy:
                accuracy = epoch_accuracy
                network = copy.deepcopy(model.state_dict())


  
    # load best model weights
    model.load_state_dict(network)
    return model