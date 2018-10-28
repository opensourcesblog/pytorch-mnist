from pytorch.main import Net
import cv2
import numpy as np
import math
from scipy import ndimage
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def getBestShift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)

    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def predict(model, device, input_images):

    # create an an array where we can store our pictures
    images = np.zeros((len(input_images), 784))
    # and the correct values
    correct_vals = np.zeros(len(input_images), dtype=int)

    # we want to test our images which you saw at the top of this page
    i = 0
    # for no in [8,0,4,3]:
    for no in input_images:

        # read the image
        gray = cv2.imread("blog/own_"+str(no)+".png", 0)
        # gray = cv2.imread(no, 0)

        # rescale it
        gray = cv2.resize(255-gray, (28, 28))
        # better black and white version
        (thresh, gray) = cv2.threshold(gray, 128,
                                       255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        while np.sum(gray[0]) == 0:
            gray = gray[1:]

        while np.sum(gray[:, 0]) == 0:
            gray = np.delete(gray, 0, 1)

        while np.sum(gray[-1]) == 0:
            gray = gray[:-1]

        while np.sum(gray[:, -1]) == 0:
            gray = np.delete(gray, -1, 1)

        rows, cols = gray.shape

        if rows > cols:
            factor = 20.0/rows
            rows = 20
            cols = int(round(cols*factor))
            # first cols than rows
            gray = cv2.resize(gray, (cols, rows))
        else:
            factor = 20.0/cols
            cols = 20
            rows = int(round(rows*factor))
            # first cols than rows
            gray = cv2.resize(gray, (cols, rows))

        colsPadding = (int(math.ceil((28-cols)/2.0)),
                       int(math.floor((28-cols)/2.0)))
        rowsPadding = (int(math.ceil((28-rows)/2.0)),
                       int(math.floor((28-rows)/2.0)))
        gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

        shiftx, shifty = getBestShift(gray)
        shifted = shift(gray, shiftx, shifty)
        gray = shifted

        # save the processed images
        cv2.imwrite("pro-img/image_"+str(no)+".png", gray)
        """
		all images in the training set have an range from 0-1
		and not from 0-255 so we divide our flatten images
		(a one dimensional vector with our 784 pixels)
		to use the same 0-1 based range
		"""
        flatten = gray.flatten() / 255.0
        """
		we need to store the flatten image and generate
		the correct_vals array
		correct_val for the first digit (9) would be
		[0,0,0,0,0,0,0,0,0,1]
		"""
        images[i] = (flatten-0.1307)/0.3081
        correct_vals[i] = no
        i += 1

    images = np.reshape(images, (len(input_images), 1, 28, 28))
    model.eval()
    test_loss = 0
    correct = 0
    images = torch.from_numpy(images).float()
    correct_vals = torch.from_numpy(correct_vals)
    data, target = images.to(device), correct_vals.to(device)
    output = model(data)
    # sum up batch loss
    test_loss += F.nll_loss(output, target, reduction='sum').item()
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(input_images)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(input_images),
        100. * correct / len(input_images)))


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def save_checkpoint(state, is_best, filename='cps/checkpoint.pth.tar'):
    torch.save(state, filename)


def main():
        # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--no_pretrained', action='store_true', default=False,
                        help="Don't use pretrained file: cps/mnist.pth.tar")
    parser.add_argument('--train', action='store_true', default=False,
                        help='Train before testing')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)

    torch.manual_seed(args.seed)

    if args.no_pretrained == False:
        checkpoint = torch.load("cps/mnist.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])

    if args.train == True:
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)

            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, True, "cps/mnist.pth.tar")

    predict(model, device, [0, 8, 4, 3])


if __name__ == '__main__':
    main()
