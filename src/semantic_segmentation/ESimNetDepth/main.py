import os
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms

import src.semantic_segmentation.ESimNetDepth.utils.transforms as ext_transforms
from src.semantic_segmentation.ESimNetDepth.models.enet import ENet, ENetDepth
from src.semantic_segmentation.ESimNetDepth.train import Train
from src.semantic_segmentation.ESimNetDepth.test import Test
from src.semantic_segmentation.ESimNetDepth.metric.iou import IoU
#from src.semantic_segmentation.ESimNetDepth.utils.args import get_arguments
from src.semantic_segmentation.ESimNetDepth.data.utils import enet_weighing, median_freq_balancing
import src.semantic_segmentation.ESimNetDepth.utils.utils as utils

from src.semantic_segmentation.ESimNetDepth.data.simnet import SimNet as dataset

# Get the arguments
#args = get_arguments()

device = torch.device('cuda')

# Mean color, standard deviation (R, G, B)
COLOR_MEAN = [0.496342, 0.466664, 0.440796]
COLOR_STD = [0.277856, 0.286230, 0.291129]

# Consts / Args
IMAGE_HEIGHT = 912
IMAGE_WIDTH = 1368
MODE = 'train'
WEIGHING = 'enet'
ARCH = 'rgbd'

# Hyperparameters
BATCH_SIZE = 12
EPOCHS = 300
LEARNING_RATE = 5e-4
BETA_0 = 0.9  # betas[0] for Adam Optimizer. Default: 0.9
BETA_1 = 0.999  # betas[1] for Adam Optimizer. Default: 0.999
LR_DECAY = 0.1  # The learning rate decay factor. Default: 0.5
LRD_EPOCHS = 100  # The number of epochs before adjusting the learning rate.
W_DECAY = 2e-4  # L2 regularization factor. Default: 2e-4

LOAD_DEPTH = True
N_WORKERS = 7
ROOT_DIR = Path('F:/data/blocks/')
SAVE_DIR = Path('F:/data/blocks/models')
NAME = 'TEST'
PRINT_STEP = 25
VALIDATE_STEP = 10

LOAD_WEIGHING = Path('F:/data/blocks/weighing.txt')


def load_dataset(dataset):
    print("\nLoading dataset...\n")

    image_transform = transforms.Compose(
        [transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
         transforms.ToTensor()])

    label_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        ext_transforms.PILToLongTensor()
    ])

    # Get selected dataset
    # Load the training set as tensors    self, root_dir, mode='train', color_mean=[0., 0., 0.], color_std=[1., 1., 1.], load_depth=True
    train_set = dataset(ROOT_DIR, mode='train', color_mean=COLOR_MEAN, color_std=COLOR_STD, load_depth=LOAD_DEPTH)
    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)

    # Load the validation set as tensors
    val_set = dataset(ROOT_DIR, mode='val', color_mean=COLOR_MEAN, color_std=COLOR_STD, load_depth=LOAD_DEPTH)
    val_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)

    # Load the test set as tensors
    test_set = dataset(ROOT_DIR, mode='test', color_mean=COLOR_MEAN, color_std=COLOR_STD, load_depth=LOAD_DEPTH)
    test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)

    # Get encoding between pixel valus in label images and RGB colors
    class_encoding = train_set.color_encoding

    # Get number of classes to predict
    num_classes = len(class_encoding)

    # Print information for debugging
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    # Get a batch of samples to display
    if MODE.lower() == 'test':
        images, labels = iter(test_loader).next()
    else:
        images, labels = iter(train_loader).next()
    print("Image size:", images.size())
    print("Label size:", labels.size())
    print("Class-color encoding:", class_encoding)

    # Show a batch of samples and labels TODO
    # if args.imshow_batch:
    #     print("Close the figure window to continue...")
    #     label_to_rgb = transforms.Compose([
    #         ext_transforms.LongTensorToRGBPIL(class_encoding),
    #         transforms.ToTensor()
    #     ])
    #     color_labels = utils.batch_transform(labels, label_to_rgb)
    #     utils.imshow_batch(images, color_labels)

    # Get class weights from the selected weighing technique
    print("\nWeighing technique:", WEIGHING)
    # If a class weight file is provided, try loading weights from in there
    class_weights = None
    if LOAD_WEIGHING.is_file():
        print('Trying to load class weights from file...')
        try:
            class_weights = np.loadtxt(LOAD_WEIGHING)
        except Exception as e:
            raise e
    if class_weights is None:
        print("Computing class weights...")
        print("(this can take a while depending on the dataset size)")
        class_weights = 0
        if WEIGHING == 'enet':
            class_weights = enet_weighing(train_loader, num_classes)
        elif WEIGHING == 'mfb':
            class_weights = median_freq_balancing(train_loader, num_classes)
        else:
            class_weights = None
        np.savetxt(LOAD_WEIGHING, class_weights)

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(device)
        # Set the weight of the unlabeled class to 0
        #if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
        class_weights[ignore_index] = 0

    print("Class weights:", class_weights)

    return (train_loader, val_loader,
            test_loader), class_weights, class_encoding


def train(train_loader, val_loader, class_weights, class_encoding):
    print("\nTraining...\n")

    num_classes = len(class_encoding)

    # Intialize ENet
    if ARCH == 'rgb':
        model = ENet(num_classes).to(device)
    elif ARCH == 'rgbd':
        model = ENetDepth(num_classes).to(device)
    else:
        raise RuntimeError('Invalid network architecture specified.')

    model = torch.nn.DataParallel(model).cuda()

    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequentely used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ENet authors used Adam as the optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA_0, BETA_1), weight_decay=W_DECAY)

    # Learning rate decay scheduler
    lr_updater = lr_scheduler.StepLR(optimizer, LRD_EPOCHS, LR_DECAY)

    # Evaluation metric
    #if args.ignore_unlabeled:
    ignore_index = list(class_encoding).index('unlabeled')
    # else:
    #     ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    # # Optionally resume from a checkpoint TODO
    # if args.resume:
    #     model, optimizer, start_epoch, best_miou = utils.load_checkpoint(
    #         model, optimizer, args.save_dir, args.name)
    #     print("Resuming from model: Start epoch = {0} "
    #           "| Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    # else:
    start_epoch = 0
    best_miou = 0

    # Start Training
    print()
    train = Train(model, train_loader, optimizer, criterion, metric, device)
    val = Test(model, val_loader, criterion, metric, device)
    for epoch in range(start_epoch, EPOCHS):
        print(">>>> [Epoch: {0:d}] Training".format(epoch))

        lr_updater.step(epoch)
        epoch_loss, (iou, miou) = train.run_epoch(PRINT_STEP)

        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
              format(epoch, epoch_loss, miou))

        if (epoch + 1) % VALIDATE_STEP == 0 or epoch + 1 == EPOCHS:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            loss, (iou, miou) = val.run_epoch(PRINT_STEP)

            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
                  format(epoch, loss, miou))

            # Print per class IoU on last epoch or if best iou
            if epoch + 1 == EPOCHS or miou > best_miou:
                for key, class_iou in zip(class_encoding.keys(), iou):
                    print("{0}: {1:.4f}".format(key, class_iou))

            # Save the model if it's the best thus far
            if miou > best_miou:
                print("\nBest model thus far. Saving...\n")
                best_miou = miou
                utils.save_checkpoint(model, optimizer, epoch + 1, best_miou,
                                      NAME, SAVE_DIR)

    return model


def test(model, test_loader, class_weights, class_encoding):
    print("\nTesting...\n")

    num_classes = len(class_encoding)

    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequentely used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Evaluation metric
    # if args.ignore_unlabeled:
    ignore_index = list(class_encoding).index('unlabeled')
    # else:
    #     ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    # Test the trained model on the test set
    test = Test(model, test_loader, criterion, metric, device)

    print(">>>> Running test dataset")

    loss, (iou, miou) = test.run_epoch(PRINT_STEP)
    class_iou = dict(zip(class_encoding.keys(), iou))

    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))

    # Print per class IoU
    for key, class_iou in zip(class_encoding.keys(), iou):
        print("{0}: {1:.4f}".format(key, class_iou))

    # Show a batch of samples and labels TODO
    # if args.imshow_batch:
    # print("A batch of predictions from the test set...")
    # images, _ = iter(test_loader).next()
    # predict(model, images, class_encoding)


def predict(model, images, class_encoding):
    images = images.to(device)

    # Make predictions!
    model.eval()
    with torch.no_grad():
        predictions = model(images)

    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it to a single int using the indices where the maximum (1) occurs
    _, predictions = torch.max(predictions.data, 1)

    label_to_rgb = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
    utils.imshow_batch(images.data.cpu(), color_predictions)


# Run only if this module is being run directly
if __name__ == '__main__':

    # Fail fast if the dataset directory doesn't exist
    assert ROOT_DIR.is_dir()

    # Fail fast if the saving directory doesn't exist
    assert SAVE_DIR.is_dir()
    #
    # # Import the requested dataset
    # if args.dataset.lower() == 'scannet':
    #     from data import ScanNet as dataset
    # else:
    #     # Should never happen...but just in case it does
    #     raise RuntimeError("\"{0}\" is not a supported dataset.".format(
    #         args.dataset))

    loaders, w_class, class_encoding = load_dataset(dataset)
    train_loader, val_loader, test_loader = loaders

    if MODE in {'train', 'full'}:
        model = train(train_loader, val_loader, w_class, class_encoding)
        if MODE == 'full':
            test(model, test_loader, w_class, class_encoding)
        print('FINITO')
    elif MODE == 'test':
        # Intialize a new ENet model
        num_classes = len(class_encoding)
        if ARCH == 'rgb':
            model = ENet(num_classes).to(device)
        elif ARCH == 'rgbd':
            model = ENetDepth(num_classes).to(device)
        else:
            # This condition will not occur (argparse will fail if an invalid option is specified)
            raise RuntimeError('Invalid network architecture specified.')

        model = torch.nn.DataParallel(model).cuda()

        # Initialize a optimizer just so we can retrieve the model from the
        # checkpoint
        optimizer = optim.Adam(model.parameters())

        # Load the previoulsy saved model state to the ENet model
        model = utils.load_checkpoint(model, optimizer, SAVE_DIR, NAME)
        print(model)
        test(model, test_loader, w_class, class_encoding)
    else:
        # Should never happen...but just in case it does
        raise RuntimeError(
            "\"{0}\" is not a valid choice for execution mode.".format(MODE))
