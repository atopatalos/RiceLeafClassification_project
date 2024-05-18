import torch
from torch import nn, optim
import numpy as np
import os
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from glob import glob
import numpy as np
import mlflow
import mlflow.pytorch


# check if cuda is available

train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
  print('CUDA is available.  Training on GPU ...')
else:
    print('CUDA is not available!  Training on CPU ...')

data_dir = './DataLabelledRice/Labelled'


images = glob(os.path.join(data_dir, '*/*.jpg'))
total_images = len(images)
print('Total images:', total_images)

# number of images per class
image_count = []
class_names = []

for folder in os.listdir(os.path.join(data_dir)):
    folder_num = len(os.listdir(os.path.join(data_dir, folder)))
    image_count.append(folder_num)
    class_names.append(folder)
    print('{:20s}'.format(folder), end=' ')
    print(folder_num)



data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(224 + 32),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

image_datasets=dict()
image_datasets['train']=datasets.ImageFolder(data_dir,transform=data_transforms['train'])
image_datasets['valid']=datasets.ImageFolder(data_dir,transform=data_transforms['valid'])

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 2
# percentage of training set to use as validation
valid_size = 0.2


# obtain training indices to use for validation
num_train = len(image_datasets['train'])
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# create dataloaders
train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
dataloaders = {}
dataloaders['train'] = train_loader
dataloaders['valid'] = valid_loader
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}


# specify class names
class_names = image_datasets['train'].classes

# download the pretrained model

model_vgg = models.vgg16(pretrained=True)
for param in model_vgg.features.parameters():
    param.requires_grad = False

n_inputs = model_vgg.classifier[6].in_features
last_layer = nn.Sequential(nn.Linear(n_inputs, 512),
                                     nn.ReLU(True),
                                     nn.BatchNorm1d(512),
                                     nn.Dropout(0.5),
                                     nn.Linear(512, 4))
model_vgg.classifier[6] = last_layer


if train_on_gpu:
    model_vgg = model_vgg.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_vgg.classifier.parameters(), lr=0.001)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_vgg.to(device);

mlflow.set_tracking_uri("http://127.0.0.1:5000")

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    # Start MLflow run
    with mlflow.start_run():
        
        for epoch in range(1, n_epochs+1):
            # initialize variables to monitor training and validation loss
            train_loss = 0.0
            valid_loss = 0.0

            # train the model
            model.train()
            for batch_idx, (data, target) in enumerate(loaders['train']):
                # move to GPU
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                ## find the loss and update the model parameters accordingly
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                ## record the average training loss
                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # validate the model
            model.eval()
            for batch_idx, (data, target) in enumerate(loaders['valid']):
                # move to GPU
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                ## update the average validation loss
                output = model(data)
                loss = criterion(output, target)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

            # log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("valid_loss", valid_loss, step=epoch)

            # print training/validation statistics
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch,
                train_loss,
                valid_loss
                ))

            ## save the model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model..'.format(valid_loss_min, valid_loss))
                # torch.save(model.state_dict(), save_path)
                mlflow.pytorch.log_model(model, "model")  # Log the model to MLflow
                valid_loss_min = valid_loss

    return model


model_vgg = train(10, dataloaders, model_vgg, optimizer,
                      criterion, train_on_gpu)