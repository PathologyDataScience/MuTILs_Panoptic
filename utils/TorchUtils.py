import os
import pickle
from warnings import warn
import torch

import MuTILs_Panoptic.utils.torchvision_transforms as tvdt


ISCUDA = torch.cuda.is_available()


def tensor_isin(arr1, arr2):
    r""" Compares a tensor element-wise with a list of possible values.
    See :func:`torch.isin`

    Source: https://github.com/pytorch/pytorch/pull/26144
    """
    result = (arr1[..., None] == arr2).any(-1)
    return result.type(torch.ByteTensor)


def transform_dlinput(
        tlist=None, make_tensor=True, flip_prob=0.5,
        augment_stain_sigma1=0.5, augment_stain_sigma2=0.5):
    """Transform input image data for a DL model.

    Parameters
    ----------
    tlist: None or list. If testing mode, pass as None.
    flip_prob
    augment_stain_sigma1
    augment_stain_sigma2

    """
    tmap = {
        'hflip': tvdt.RandomHorizontalFlip(prob=flip_prob),
        'augment_stain': tvdt.RandomHEStain(
            sigma1=augment_stain_sigma1, sigma2=augment_stain_sigma2),
    }
    tlist = [] if tlist is None else tlist
    transforms = []
    # go through various transforms
    for tname in tlist:
        transforms.append(tmap[tname])
    # maybe convert to tensor
    if make_tensor:
        # transforms.append(tvdt.PILToTensor(float16=ISCUDA))
        transforms.append(tvdt.PILToTensor(float16=False))
    return tvdt.Compose(transforms)


def load_torch_model(checkpoint_path, model, optimizer=None):
    """
    Source https://towardsdatascience.com/how-to-save-and-load-a-model-in- ...
    ... pytorch-with-a-complete-example-c2920e617dee

    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    if torch.cuda.is_available():
        extra = {}
    else:
        extra = {
            'map_location': lambda storage, loc: storage,
            # 'map_location': {'cuda:0': 'cpu'},
        }

    # load model state
    try:
        model.module.load_state_dict(torch.load(checkpoint_path, weights_only=True, **extra))
    except AttributeError:  # this is not a Dataparallel model
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True, **extra))

    # load optimizer state
    if optimizer is not None:
        optimizer_path = checkpoint_path.replace('.pt', '.optim')
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True, **extra))
        else:
            warn(f"Existing optimizer not found at: {optimizer_path}")

    to_return = {
        'model': model,
        'optimizer': optimizer,
    }

    # extra metadata (eg epoch)
    metapath = checkpoint_path.replace('.pt', '.meta')
    if os.path.exists(metapath):
        try:
            meta = pickle.load(open(metapath, 'rb'))
        except:
            meta = {'epoch': 0}
        to_return.update(meta)

    return to_return


def get_optimizer(model, optimizer_type='SGD', optimizer_params=None):
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_type == 'SGD':
        optimizer_params = {
            'lr': 0.005,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        } if optimizer_params is None else optimizer_params
        optimizer = torch.optim.SGD(params, **optimizer_params)
    elif optimizer_type == 'Adam':
        optimizer_params = {
            'lr': 1e-4,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.0001,
        } if optimizer_params is None else optimizer_params
        optimizer = torch.optim.Adam(params, **optimizer_params)
    else:
        raise NotImplementedError(f'Unknown optimizer: {optimizer_type}')
    return optimizer


def t2np(x: torch.Tensor):
    return x.detach().cpu().numpy()
