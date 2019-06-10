import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os


def batch_transform(batch, transform):
    """Applies a transform to a batch of samples.

    Keyword arguments:
    - batch (): a batch os samples
    - transform (callable): A function/transform to apply to ``batch``

    """

    # Convert the single channel label to RGB in tensor form
    # 1. torch.unbind removes the 0-dimension of "labels" and returns a tuple of
    # all slices along that dimension
    # 2. the transform is applied to each slice
    transf_slices = [transform(tensor) for tensor in torch.unbind(batch)]

    return torch.stack(transf_slices)


def imshow_batch(images, labels):
    """Displays two grids of images. The top grid displays ``images``
    and the bottom grid ``labels``

    Keyword arguments:
    - images (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)
    - labels (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)

    """

    # Make a grid with the images and labels and convert it to numpy
    images = torchvision.utils.make_grid(images).numpy()
    labels = torchvision.utils.make_grid(labels).numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
    ax1.imshow(np.transpose(images, (1, 2, 0)))
    ax2.imshow(np.transpose(labels, (1, 2, 0)))

    plt.show()


def save_checkpoint(model, optimizer, epoch, miou, name, save_dir):
    """Saves the model in a specified directory with a specified name.save

    Keyword arguments:
    - model (``nn.Module``): The model to save.
    - optimizer (``torch.optim``): The optimizer state to save.
    - epoch (``int``): The current epoch for the model.
    - miou (``float``): The mean IoU obtained by the model.
    - args (``ArgumentParser``): An instance of ArgumentParser which contains
    the arguments used to train ``model``. The arguments are written to a text
    file in ``args.save_dir`` named "``args.name``_args.txt".

    """
    assert save_dir.is_dir(), "The directory \"{0}\" doesn't exist.".format(str(save_dir))

    # Save model
    model_path = save_dir / name
    torch.save(model.state_dict(), model_path)

    # Save arguments
    summary_filename = save_dir / (name + '_summary.txt')
    with open(summary_filename, 'w') as summary_file:
        # TODO:
        # sorted_args = sorted(vars(args))
        # summary_file.write("ARGUMENTS\n")
        # for arg in sorted_args:
        #     arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
        #     summary_file.write(arg_str)

        summary_file.write("\nBEST VALIDATION\n")
        summary_file.write("Epoch: {0}\n". format(epoch))
        summary_file.write("Mean IoU: {0}\n". format(miou))


def load_checkpoint(model, optimizer, folder_dir, filename):
    """Saves the model in a specified directory with a specified name.save

    Keyword arguments:
    - model (``nn.Module``): The stored model state is copied to this model
    instance.
    - optimizer (``torch.optim``): The stored optimizer state is copied to this
    optimizer instance.
    - folder_dir (``string``): The path to the folder where the saved model
    state is located.
    - filename (``string``): The model filename.

    Returns:
    The epoch, mean IoU, ``model``, and ``optimizer`` loaded from the
    checkpoint.

    """
    assert folder_dir.is_dir(), "The directory \"{0}\" doesn't exist.".format(str(folder_dir))

    # Create folder to save model and information
    model_path = folder_dir / filename
    assert model_path.is_file(), "The model file \"{0}\" doesn't exist.".format(str(filename))

    # Load the stored model parameters to the model instance
    #checkpoint =
    model.load_state_dict(torch.load(model_path))
    #optimizer.load_state_dict(checkpoint['optimizer'])
    #epoch = checkpoint['epoch']
    #miou = checkpoint['miou']

    return model  #, optimizer, epoch, miou
