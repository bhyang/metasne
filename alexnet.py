import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import torch.utils.trainer as trainer
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.trainer.plugins
from tensorboard_logger import configure, log_value

import numpy as np

import os

# Data Loading
transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ]),
])

traindir = '/media/4tb/owens/ILSVRC2012/train/'
train = datasets.ImageFolder(traindir, transform)
train_loader = torch.utils.data.DataLoader(
    train, batch_size=64, shuffle=True, num_workers=2)

model = models.alexnet(pretrained=True).cuda()
criterion = nn.CrossEntropyLoss().cuda()
lr = 1e-2
iter_counter = 0

print('Initializing peturbances...')
for param in model.parameters():
    param.data += np.random.normal(0, 1e-3)

print('Beginning training...')
configure('runs/lr3', flush_secs=5)
while iter_counter < 5000:
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        if iter_counter % 500 == 0:
            torch.save(model.state_dict(), '1_sd_' + str(iter_counter))
        if iter_counter >= 5000:
            break
        if iter_counter % 1000 == 0:
            lr *= .1
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9)
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        curr_loss = loss.data[0]
        print('iter ' + str(iter_counter) + ': ' + str(curr_loss))
        log_value('loss', curr_loss, iter_counter)
        running_loss += curr_loss
        # if i % 1000 == 999:    # print every 1000 mini-batches
        #     print('[%5d] loss: %.3f' %
        #           (i + 1, running_loss / 1000))
        running_loss = 0.0

        iter_counter += 1

print('Finished Training')
