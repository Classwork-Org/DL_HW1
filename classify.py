import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import sys

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        # first hidden layer, input is specific to CIFAR10 dataset image size
        self.hidden1 = nn.Linear(3 * 32 * 32, 1000)
        # series of identical linear layers, sizes were originally going to be 
        # passed at training time but I got lucky and found a good sized network
        # right away
        self.hidden_list = nn.ModuleList(
            [nn.Linear(1000, 1000) for i in range(2)])
        # output layer, output size = number of class labels
        self.out = nn.Linear(1000, 10)
        # set dropout layer with dropout mask at 10% of input
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # flatten input image from 3 channel to vector
        x = x.view(-1, 3 * 32 * 32)
        # apply first hidden layer with relu activation
        x = self.hidden1(x)
        x = F.relu(x)
        # apply random dropout mask
        x = self.dropout(x)
        # repeat for regular layers
        for _, layer in enumerate(self.hidden_list):
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
        # evaluate output, no need to add softmax here because loss_fn
        # (CrossEntropy) already applies log(softmax)
        # all it will do here is scale output but max will remain the same
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        outp = (self.out(x))
        return outp


def train():

    BATCH_SIZE = 64

    #transform input to tensor
    transform = transforms.Compose(
        [transforms.ToTensor()])

    train_data = torchvision.datasets.CIFAR10(
        root='./data.cifar10',  # location of the dataset
        train=True,  # this is training data
        # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W)
        transform=transform,
        download=True  # if you haven't had the dataset, this will automatically download it for you
    )

    #setup dataloader for training set
    train_loader = Data.DataLoader(dataset=train_data,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=8, pin_memory=True)

    test_data = torchvision.datasets.CIFAR10(
        root='./data.cifar10/',
        train=False,
        transform=transform)

    #setup dataloader for test set
    test_loader = Data.DataLoader(dataset=test_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=8, pin_memory=True)

    #create model
    model = NN()
    
    #setup optimizer
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.7, lr=0.01)
    loss_func = torch.nn.CrossEntropyLoss()
    
    #move model parameters to gpu
    model = model.cuda()

    for epoch in range(21):
        # setup running loss for mini-batch group
        running_loss = 0.0
        for step, (input, target) in enumerate(train_loader):
            # get batch from train loader and move to gpu
            input = input.cuda()
            target = target.cuda()
            # train w optimizer
            model.train()
            # get output
            output = model(input)
            # evaluate loss
            loss = loss_func(output, target)
            # zero out accumulated gradients
            optimizer.zero_grad()
            # evaluate gradients
            loss.backward()
            # updated weights
            optimizer.step()
            # accumulate loss for this group of mini batches (200)
            running_loss += loss.item()
            # every 200 mini batches stop and evaluate accuracy on test set
            if step != 0 and step % 200 == 0:   
                # switch model to eval mode to disable dropout layer
                model.eval()
                ## test_loss = 0
                correct = 0
                # get mini batches from test loader
                for data, target in test_loader:
                    # move data and target to gpu
                    data = data.cuda()
                    target = target.cuda()
                    # do inference pass
                    output = model(data)
                    # get the index of the max log-probability
                    pred = output.data.max(1, keepdim=True)[1]
                    # calculate number of correct predictions
                    correct += pred.eq(target.data.view_as(pred)).sum()
                    ## test_loss += nn.CrossEntropyLoss()(output, target)

                # print mini batch group results on test set
                print(
                    '\nTest set: Epoch [{}]:Step[{}] Training loss: {:.4f}, Test Accuracy: ({:.3f}%)'
                    .format(
                        epoch, step, running_loss, float(100*correct) / float(len(test_loader.dataset))))

                # reset loss for next mini batch group
                running_loss = 0.0

    # after 20 Epochs save model (No early stopping)
    torch.save(model.state_dict(), './model/CIFAR10_NN1')

    return model


def test(modelPath, imagePath):
    # print("Loading pretrained CIFAR10 model")
    # setup cifar class tuple (Order of labels taken from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # create model
    model = NN()
    # attempt to load model state dict based on supplied path
    try:
        model_dict = torch.load(modelPath)
    except:
        print("Failed to load model. Please run script in train mode first before inference")
        exit(-1)
    
    # load weights into model
    model.load_state_dict(model_dict)
    ## print("Model loaded")
    # switch model to eval mode to disable dropout during inference
    model.eval()
    # load image
    try:
        image = Image.open(imagePath)
    except:
        print("Failed to load supplied image. Please check image path")
        exit(-1)
    
    # convert loaded image to tensor
    x = TF.to_tensor(image)
    #remove alpha channel if present
    x = x[:3, :, :] 
    # add batch dim
    x = torch.unsqueeze(x, 0)
    # down/up sample image to network input
    x = F.interpolate(x, (32, 32)) 
    ## print("Classifying")
    # run inference pass, softmax just scales output to 0-1, argmax will remain unchanged
    outputs = F.softmax(model(x), dim=1)
    # get highest class loglikelyhood
    predicted = torch.argmax(outputs)
    print("Predicted: {}".format(classes[predicted]))

if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print("Please specify mode (train/test)")
        exit(-1)
    if(sys.argv[1] == 'test'):
        if(len(sys.argv) < 3):
            print("Please specify image path to classify")
            exit(-1)
        imagePath = sys.argv[2]
        test('./model/CIFAR10_NN1', imagePath)
    elif(sys.argv[1] == 'train'):
        train()
    else:
        print("Invalid mode specified. Please specify mode (train/test)")
        exit(-1)        