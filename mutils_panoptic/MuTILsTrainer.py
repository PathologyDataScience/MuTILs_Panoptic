import os
from os.path import join as opj
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import MultiStepLR
from pandas import DataFrame, read_csv
import torch
from torch import nn
import numpy as np
import pickle
from scipy.stats import pearsonr, spearmanr

# command line interface
import argparse
from MuTILs_Panoptic.utils.GeneralUtils import maybe_mkdir, AllocateGPU, rmse

parser = argparse.ArgumentParser(description='Run mutils model.')
parser.add_argument('-f', type=int, default=1, help='fold')
parser.add_argument('-g', type=int, default=[0], nargs='+', help='gpu(s)')
parser.add_argument('--dep', type=int, default=1, help='default start epoch.')
args = parser.parse_args()

# GPU allocation MUST happen before importing other modules
AllocateGPU(GPUs_to_use=args.g)

# own imports
from MuTILs_Panoptic.utils.MiscRegionUtils import _aggregate_semsegm_stats
from MuTILs_Panoptic.utils.RegionPlottingUtils import (
    vis_mutils_inference, plot_batch_losses, plot_eval_metrics
)
from MuTILs_Panoptic.mutils_panoptic.RegionDatasetLoaders import (
    get_cv_fold_slides, MuTILsDataset
)
from MuTILs_Panoptic.mutils_panoptic.MuTILs import (
    MuTILs, MuTILsLoss, MuTILsEvaluator
)
from MuTILs_Panoptic.utils.TorchUtils import get_optimizer, load_torch_model

# eval if cpu vs gpu
ISCUDA = torch.cuda.is_available()
if ISCUDA:
    try:
        NGPUS = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    except KeyError:  # gpus available but not assigned
        NGPUS = 8  # NU system
else:
    NGPUS = 0

# =============================================================================


@torch.no_grad()
def evaluate_mutils(
        model: MuTILs, test_loader: DataLoader, savedir: str,
        epoch: int, printfreq=5, vis=True, visfreq=10):
    """
    Evaluate MuTILs model.
    """
    # set to eval mode
    model.eval()

    device = torch.device('cuda') if ISCUDA else torch.device('cpu')
    evaluator = MuTILsEvaluator()
    all_stats = []
    batchidx = 0
    nbatches = len(test_loader)

    # batchwise inference
    for batchdata, truth in test_loader:
        batchidx += 1

        # monitor
        batchstr = f'Epoch {epoch}: Evaluator: batch {batchidx} of {nbatches}'
        if (batchidx - 1) % printfreq == 0:
            print(batchstr)

        # move tensors to right device
        ignore = ['idx', 'roiname']
        batchdata = [
            {k: v.to(device) for k, v in bd.items()}
            for bd in batchdata]
        truth = [{
            k: v.to(device) if k not in ignore else v for k, v in
            bd.items()}  # noqa
            for bd in truth]

        # do inference
        inference = model(batchdata)

        # visualize to monitor progress
        if vis and ((batchidx - 1) % visfreq == 0):
            print(f'{batchstr}: visualizing ...')
            try:
                vis_mutils_inference(
                    batchdata=batchdata, inference=inference, truth=truth,
                    savename=opj(
                        savedir, 'vis', f'batch-{batchidx}_epoch-{epoch}'))
            except Exception as problem:
                print(f'{batchstr}: Ooops! Visualization went wrong!')
                print(problem.__repr__())

        # calculate stats
        batch_stats = evaluator(inference=inference, truth=truth)
        all_stats.extend(batch_stats)

    # convert all to df
    all_stats = DataFrame.from_records(all_stats)

    # add meta
    # noinspection PyUnresolvedReferences
    all_stats.loc[:, 'slide'] = all_stats.loc[:, 'roiname'].map(
        test_loader.dataset._r2s)

    # aggregate tiles from the same slide & save for later
    agg_stats = []
    slides = np.unique(all_stats.loc[:, 'slide'])
    for slide in slides:
        agg = _aggregate_semsegm_stats(
            df=all_stats.loc[all_stats.loc[:, 'slide'] == slide, :])
        agg_stats.append(agg)
    agg_stats = DataFrame.from_records(agg_stats)
    cols = list(agg_stats.columns)
    agg_stats.loc[:, 'slide'] = slides
    agg_stats.loc[:, 'epoch'] = epoch
    agg_stats = agg_stats.loc[:, ['slide', 'epoch'] + cols]
    agg_stats.to_csv(opj(
        savedir, 'metrics', f'testing_metrics_epoch-{epoch}.csv'))

    # get aggregate stats over whole set -- mean
    stdf_mean = DataFrame(agg_stats.mean(axis=0)).T  # this ignores nans.

    # get aggregate stats over whole set -- std
    stdf_std = DataFrame(agg_stats.std(axis=0)).T  # this ignores nans.
    stdf_std.loc[:, 'epoch'] = epoch

    # add RMSE and correl. for TILs score
    for pf in ['roi', 'hpf']:
        pfx = f'{pf}-CTA-score'
        stdf_mean[f'{pfx}_rmse'] = \
            rmse(agg_stats.loc[:, f'{pfx}_pred'], agg_stats.loc[:, f'{pfx}_true'])  # noqa
        try:
            stdf_mean[f'{pfx}_pearson_r'], stdf_mean[f'{pfx}_pearson_pval'] = \
                pearsonr(agg_stats.loc[:, f'{pfx}_pred'], agg_stats.loc[:, f'{pfx}_true'])  # noqa
            stdf_mean[f'{pfx}_spearman_r'], stdf_mean[f'{pfx}_spearman_pval'] = \
                spearmanr(agg_stats.loc[:, f'{pfx}_pred'], agg_stats.loc[:, f'{pfx}_true'])  # noqa
        except Exception as problem:
            print(problem.__repr__())

    # save
    savename = opj(savedir, 'metrics', 'testing_metrics_mean.csv')
    stdf_mean.to_csv(
        savename, mode='a' if os.path.exists(savename) else 'w',
        header=not os.path.exists(savename))
    savename = opj(savedir, 'metrics', 'testing_metrics_std.csv')
    stdf_std.to_csv(
        savename, mode='a' if os.path.exists(savename) else 'w',
        header=not os.path.exists(savename))


def _train_one_mutils_epoch(
        model: MuTILs, optimizer, train_loader: DataLoader,
        criterion: MuTILsLoss, device, epoch: int, printfreq: int,
        savedir: str, ckpt_path: str):
    """Train one epoch. Internal use."""

    # set to training mode
    model.train()

    batch_losses = []
    batchidx = 0

    for batchdata, truth in train_loader:

        batchidx += 1

        # move tensors to right device
        ignore = ['idx', 'roiname']
        batchdata = [
            {k: v.to(device) for k, v in bd.items()}
            for bd in batchdata]
        truth = [{
            k: v.to(device) if k not in ignore else v for k, v in
            bd.items()}  # noqa
            for bd in truth]

        # run in forward mode (train)
        inference = model(batchdata)

        # calculate loss
        loss_dict = criterion(inference=inference, truth=truth)

        # Register loss
        simple_losses = {'epoch': epoch, 'batch': batchidx}
        simple_losses.update({k: float(v) for k, v in loss_dict.items()})
        if (batchidx - 1) % printfreq == 0:
            print({k: '%.3f' % v for k, v in simple_losses.items()})
        batch_losses.append(simple_losses)

        # backpropagate
        optimizer.zero_grad()
        loss_dict['all'].backward()
        optimizer.step()

    # save loss
    savename = opj(savedir, 'metrics', 'training_loss.csv')
    batch_ldf = DataFrame.from_records(batch_losses)
    batch_ldf.to_csv(
        savename, mode='a' if os.path.exists(savename) else 'w',
        header=not os.path.exists(savename),
    )

    # save checkpoint
    print("*--- SAVING CHECKPOINT!! ---*")
    try:
        torch.save(model.module.state_dict(), f=ckpt_path)
    except AttributeError:
        torch.save(model.state_dict(), f=ckpt_path)
    torch.save(
        optimizer.state_dict(),
        f=ckpt_path.replace('.pt', '.optim'))
    meta = {'epoch': epoch}
    with open(ckpt_path.replace('.pt', '.meta'), 'wb') as f:
        pickle.dump(meta, f)


# noinspection PyShadowingNames
def train_mutils_fold(
        basedir: str, cfg, fold: int, n_grupdates: int, smooth_window=20,
        printfreq=10, batch_visfreq=10, epoch_visfreq=10,
        _default_epoch=1, _train=True, _evaluate=True):
    """"""

    mtp = cfg.MuTILsParams

    # GPU vs CPU
    device = torch.device('cuda') if ISCUDA else torch.device('cpu')

    # We assume that the base folder name is the model name
    model_name = os.path.basename(basedir)

    # where to save stuff
    maybe_mkdir(basedir)
    savedir = opj(basedir, f'fold_{fold}')
    maybe_mkdir(savedir)
    maybe_mkdir(opj(savedir, 'metrics'))
    maybe_mkdir(opj(savedir, 'vis'))

    # load train/test slides for fold
    train_slides, test_slides = get_cv_fold_slides(
        train_test_splits_path=opj(mtp.root, 'train_test_splits'), fold=fold)

    # Training dataset loader
    train_dataset = MuTILsDataset(
        root=mtp.root, slides=train_slides, **mtp.train_dataset_params)
    mtp.train_loader_kwa['sampler'] = WeightedRandomSampler(
        weights=train_dataset.roi_weights.values,
        num_samples=len(train_dataset.roi_weights),
        replacement=True,
    )
    train_loader = DataLoader(dataset=train_dataset, **mtp.train_loader_kwa)
    n_batches = len(train_loader)
    assert smooth_window <= n_batches

    # Testing dataset loader
    test_dataset = MuTILsDataset(
        root=mtp.root, slides=test_slides, **mtp.test_dataset_params)
    test_loader = DataLoader(dataset=test_dataset, **mtp.test_loader_kwa)

    # init model
    model = MuTILs(**mtp.model_params)

    # maybe use half floats
    if train_dataset.float16:
        model = model.to(torch.float16)

    # define loss calculation
    criterion = MuTILsLoss(**mtp.loss_params)

    # move all to the right device.
    # This must be done BEFORE constructing the optimizer!
    model.to(device)
    criterion.to(device)

    # GPU parallelize if mutiple gpus are available. See:
    #   https://pytorch.org/tutorials/beginner/blitz/ ...
    #   data_parallel_tutorial.html#create-model-and-dataparallel
    if NGPUS > 1:
        print(f"Let's use {NGPUS} GPUs!")
        model = nn.DataParallel(model)

    # construct an optimizer
    optimizer = get_optimizer(model=model, **mtp.optimizer_kws)

    # load weights and optimizer state
    ckpt_path = opj(savedir, f'{model_name}.pt')
    if os.path.exists(ckpt_path):
        ckpt = load_torch_model(ckpt_path, model=model, optimizer=optimizer)
        model = ckpt['model']
        optimizer = ckpt['optimizer']
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch']
            start_epoch += 1
        else:
            start_epoch = _default_epoch
    else:
        start_epoch = _default_epoch

    # internal-external cross-val means n batches varies by fold
    n_epochs = int(np.ceil(n_grupdates / n_batches))

    # init LR scheduler
    if start_epoch > 1:
        mtp.lr_scheduler_kws['last_epoch'] = start_epoch - 1
    try:
        lr_scheduler = MultiStepLR(optimizer, **mtp.lr_scheduler_kws)

    except KeyError:
        # This occurs when I idiotically delete the opimizer but later decide I
        # want to continue training my model. See:
        #   https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
        mtp.lr_scheduler_kws['last_epoch'] = -1
        lr_scheduler = MultiStepLR(optimizer, **mtp.lr_scheduler_kws)
        for _ in range(_default_epoch):
            lr_scheduler.step()

    # iterate through epochs
    for epoch in range(start_epoch, n_epochs + 1):

        # train epoch
        if _train:
            _train_one_mutils_epoch(
                model=model, optimizer=optimizer, train_loader=train_loader,
                criterion=criterion, device=device, epoch=epoch,
                printfreq=printfreq, savedir=savedir, ckpt_path=ckpt_path)

        # evaluate model on testing set
        if _evaluate:
            evaluate_mutils(
                model=model, test_loader=test_loader, savedir=savedir,
                epoch=epoch, printfreq=printfreq, visfreq=batch_visfreq,
                vis=any([
                    epoch == 1,
                    epoch == n_epochs,
                    epoch % epoch_visfreq == 0
                ]),
            )

        # maybe update learning rate
        lr_scheduler.step()

        # plot training losses
        savename = opj(savedir, 'metrics', 'training_loss')
        batch_ldf = read_csv(savename + '.csv', index_col=0)
        batch_ldf.loc[:, 'batch'] = np.arange(1, batch_ldf.shape[0] + 1)
        plot_batch_losses(
            loss_df=batch_ldf, savename=savename, title='All epochs',
            window=smooth_window)

        # plot epoch-by-epoch testing accuracies
        savename = opj(savedir, 'metrics', 'testing_metrics')
        plot_eval_metrics(
            means=read_csv(savename + '_mean.csv', index_col=0),
            stds=read_csv(savename + '_std.csv', index_col=0),
            savename=savename)


# =============================================================================

if __name__ == '__main__':

    from MuTILs_Panoptic.utils.MiscRegionUtils import load_region_configs
    from MuTILs_Panoptic.utils.GeneralUtils import save_configs

    # paths
    BASEPATH = opj(os.path.expanduser('~'), 'Desktop', 'cTME')
    model_name = 'mutils_06022021'
    model_root = opj(BASEPATH, 'results', 'mutils', 'models', model_name)
    maybe_mkdir(model_root)

    # load configs
    configs_path = opj(model_root, 'region_model_configs.py')
    cfg = load_region_configs(configs_path=configs_path)

    # for reproducibility, copy configs & most relevant code file to results
    if not os.path.exists(configs_path):
        save_configs(
            configs_path=opj(
                BASEPATH, 'ctme', 'configs', 'region_model_configs.py'),
            results_path=model_root)
    save_configs(
        configs_path=os.path.abspath(__file__),
        results_path=model_root, warn=False)
    save_configs(
        configs_path=opj(BASEPATH, 'ctme', 'regions', 'MuTILs.py'),
        results_path=model_root, warn=False)

    # train
    if args.f == 999:
        train_mutils_fold(
            basedir=model_root, cfg=cfg, fold=args.f, n_grupdates=50000,
            smooth_window=2, printfreq=2, batch_visfreq=2, epoch_visfreq=1,
            # # debug mode!!
            # _train=False, _evaluate=True,
        )
    else:
        train_mutils_fold(
            basedir=model_root, cfg=cfg, fold=args.f, n_grupdates=150000,
            smooth_window=64, printfreq=10, batch_visfreq=10,
            epoch_visfreq=20, _default_epoch=args.dep,

            # tmp debug mode!!!
            # smooth_window=2, printfreq=1, batch_visfreq=10000,
            # epoch_visfreq=1, _default_epoch=args.dep,
            # _train=False, _evaluate=True,
        )