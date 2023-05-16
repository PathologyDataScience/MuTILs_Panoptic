import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from pandas import DataFrame
import PIL
from io import BytesIO

from MuTILs_Panoptic.configs.panoptic_model_configs import (
    RegionCellCombination as rcc, VisConfigs
)


def get_visualization_ready_combined_mask(mask, issuper=False):
    """
    Given a mask where first channel is region semantic segmentation mask,
    second is nuclei semantic segmentation mask, and third is the nucl. edges,
    parse a single-channel mask that can be visualized, containing nuclei
    superimposed on regions with delineated nuclear boundaries.
    """
    msk = mask.copy()
    # background (non-nuclear) maps to zero (overwritten by regions)
    bckg = rcc.SUPERNUCLEUS_CODES['BACKGROUND'] if issuper \
        else rcc.NUCLEUS_CODES['BACKGROUND']
    msk[..., 1][msk[..., 1] == bckg] = 0
    # superimpose nuclei on regions
    msk[..., 1][msk[..., 1] > 0] += len(VisConfigs.REGION_COLORS) - 1
    msk[..., 0][msk[..., 1] > 0] = 0
    msk[..., 0] = msk[..., 0] + msk[..., 1]
    # black nuclear edges
    if mask.shape[2] == 3:
        msk[..., 0][msk[..., 2] > 0] = 0
    # make sure all classes are represented for vis
    ncols = len(VisConfigs.SUPERCOMBINED_COLORS) if issuper \
        else len(VisConfigs.COMBINED_COLORS)
    msk[np.arange(ncols), 0, 0] = np.arange(ncols)

    return msk[..., 0]


def save_combined_mask_vs_rgb_visualization(rgb, mask, savename, fovloc=None):
    """"""
    vismask = get_visualization_ready_combined_mask(mask)

    fheight = 5
    fwidth = 10
    fig, ax = plt.subplots(1, 2, figsize=(fwidth, fheight), sharey=True)
    ax[0].imshow(rgb)
    ax[1].imshow(vismask, cmap=VisConfigs.COMBINED_CMAP)
    if fovloc is not None:
        for axis in ax:
            axis.add_patch(Rectangle(
                xy=(fovloc['left'], fovloc['top']),
                width=fovloc['right'] - fovloc['left'],
                height=fovloc['bottom'] - fovloc['top'],
                fill=False, edgecolor='yellow', linewidth=1.5,
            ))
    ax[0].axis('off')
    ax[1].axis('off')
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()


def get_rgb_visualization_from_mask(combined_mask):
    """"""
    mask = get_visualization_ready_combined_mask(combined_mask, issuper=False)

    # later on flipped by matplotlib for weird reason
    mask = np.flipud(mask)

    fig = plt.figure(
        figsize=(mask.shape[1] / 1000, mask.shape[0] / 1000), dpi=100)
    ax = plt.subplot(111)
    ax.imshow(mask, cmap=VisConfigs.COMBINED_CMAP, interpolation='nearest')

    plt.axis('off')
    ax = plt.gca()
    ax.set_xlim(0.0, mask.shape[1])
    ax.set_ylim(0.0, mask.shape[0])

    ax.axis('off')
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)

    buf = BytesIO()
    plt.savefig(buf, format='png', pad_inches=0, dpi=1000)
    buf.seek(0)
    rgb_vis = np.uint8(PIL.Image.open(buf))[..., :3]
    plt.close()

    return rgb_vis


# noinspection LongLine,DuplicatedCode
def vis_mutils_inference(
        batchdata: dict, inference: dict, truth: dict, savename: str,
        norm_cmap=False, cmap='viridis'):
    """"""
    roinamestr = ""
    nrs = len(inference['hpf_roidx'])
    fheight = 3.67 * nrs
    fwidth = 3.67 * 7  # 7 panels

    fig, ax = plt.subplots(nrs, 7, figsize=(fwidth, fheight))

    # mFIXME: The only visualized the first hpf per roi

    for rno in range(nrs):

        roinamestr += truth[rno]['roiname'] + '\n'

        pred_hpfloc = inference['hpf_hres_bounds'][rno, ...]
        xmin, ymin, xmax, ymax = [int(j) for j in pred_hpfloc]

        rgb = np.uint8(batchdata[rno]['highres_rgb'][0, ...].cpu() * 255.)
        rgb = rgb.transpose(1, 2, 0)
        ax[rno, 0].imshow(rgb)
        ax[rno, 0].add_patch(Rectangle(
            xy=(xmin, ymin), width=xmax - xmin, height=ymax - ymin,
            fill=False, edgecolor='k', linewidth=2.5,
        ))
        ax[rno, 0].set_title('rgb')

        mask = np.uint8(truth[rno]['highres_mask'][0, ...].cpu())
        mask = mask.transpose(1, 2, 0)
        vis_mask = get_visualization_ready_combined_mask(mask.copy())
        ax[rno, 1].imshow(vis_mask, cmap=VisConfigs.COMBINED_CMAP,
                          interpolation='nearest')
        ax[rno, 1].set_title('true mask')

        roi_rpred = inference['roi_region_logits'][
            rno, ...].detach().cpu().numpy()
        roi_rpred = np.argmax(roi_rpred, 0) + 1
        roi_rpred[0, :9] = np.arange(9)
        ax[rno, 2].imshow(roi_rpred, cmap=VisConfigs.REGION_CMAP,
                          interpolation='nearest')
        ax[rno, 2].set_title('roi pred')

        sal_pred = inference['roi_saliency_matrix'][
            rno, ...].detach().cpu().numpy()
        kw = {
            'cmap': cmap,
            'cbar': False,
            'linewidths': 0.5,
            'linecolor': 'k',
            # 'annot': False,
            'annot': True,
            'fmt': '.2f',
            'annot_kws': {'fontsize': 12},
        }
        if norm_cmap:
            kw.update({'vmin': 0., 'vmax': 1.})
        ax[rno, 3] = sns.heatmap(sal_pred, ax=ax[rno, 3], **kw)
        ax[rno, 3].set_title('sal pred')

        hpf_rpred = inference['hpf_region_logits'][rno, :, :,
                    :].detach().cpu().numpy()  # noqa
        hpf_rpred = np.argmax(hpf_rpred, 0) + 1
        hpf_rpred[0, :9] = np.arange(9)
        ax[rno, 4].imshow(hpf_rpred, cmap=VisConfigs.REGION_CMAP,
                          interpolation='nearest')
        ax[rno, 4].set_title('hpf region pred')

        hpf_npred1 = inference['hpf_nuclei_pre'][rno, :, :,
                     :].detach().cpu().numpy()
        hpf_npred1 = np.argmax(hpf_npred1, 0) + 1
        hpf_npred1[0, :6] = np.arange(6)
        ax[rno, 5].imshow(hpf_npred1, cmap=VisConfigs.NUCLEUS_CMAP,
                          interpolation='nearest')
        ax[rno, 5].set_title('hpf nuclei pred (pre)')

        hpf_npred2 = inference['hpf_nuclei'][rno, :, :,
                     :].detach().cpu().numpy()
        hpf_npred2 = np.argmax(hpf_npred2, 0) + 1
        hpf_npred2[0, :6] = np.arange(6)
        ax[rno, 6].imshow(hpf_npred2, cmap=VisConfigs.NUCLEUS_CMAP,
                          interpolation='nearest')
        ax[rno, 6].set_title('hpf nuclei pred (post)')

    plt.tight_layout(pad=0.)
    plt.savefig(savename + '.png')
    plt.close()

    # make sure you know which images are visualized
    with open(savename + '.txt', 'w') as f:
        f.write(roinamestr)


# noinspection DuplicatedCode
def plot_batch_losses(loss_df: DataFrame, savename: str, window=0, title=''):
    """"""
    loss_df.reset_index(inplace=True)

    # maybe smooth losses and subsample
    if window > 0:
        carryover = ['epoch', 'batch']
        smoothed = loss_df.rolling(window=window).mean()
        smoothed.loc[:, carryover] = loss_df.loc[:, carryover]
        idxs = np.arange(window, smoothed.shape[0], window)
        smoothed = smoothed.iloc[smoothed.index[idxs], :]
    else:
        smoothed = loss_df

    plt.figure(figsize=(7, 7))
    lcols = {
        'roi_regions': 'purple',
        'hpf_regions': 'magenta',
        'hpf_nuclei_pre': 'cornflowerblue',
        'hpf_nuclei': 'blue',
        'all': 'k',
    }
    lcols = {k: v for k, v in lcols.items() if k in smoothed.columns}
    for lname, lcol in lcols.items():
        plt.plot(
            smoothed.loc[:, 'batch'],
            smoothed.loc[:, lname],
            color=lcol,
            label=lname,
            linewidth=1.5,
        )
    plt.xlabel('gradient update', fontsize=11)
    plt.ylabel('loss value', fontsize=11)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    # plt.show()
    plt.savefig(savename + '.svg')
    plt.close()


# noinspection LongLine,DuplicatedCode
def plot_eval_metrics(means: DataFrame, stds: DataFrame, savename: str):
    """"""

    # noinspection PyShadowingNames,LongLine
    def _plot_ax(ax, col, label, fill=True, stderr=True):
        x = means.loc[:, 'epoch']
        y = means.loc[:, col]
        if fill:
            pm = stds.loc[:, col]
            if stderr:
                pm = pm / np.sqrt(pm.shape[0])
            ax.fill_between(x, y - pm, y + pm, facecolor=mcolors[col],
                            alpha=0.1)
        ax.plot(x, y, color=mcolors[col], label=label, linewidth=2.0)
        return ax

    fig, axes = plt.subplots(1, 4, figsize=(5 * 4, 5))

    mcolors = {
        'roi-regions_TUMOR-segm_dice': VisConfigs.REGION_COLORS['TUMOR'],
        'roi-regions_STROMA-segm_dice': VisConfigs.REGION_COLORS['STROMA'],
        'roi-regions_TILS-segm_dice': VisConfigs.REGION_COLORS['TILS'],
        # 'hpf-regions_TUMOR-segm_dice': VisConfigs.REGION_COLORS['TUMOR'],
        # 'hpf-regions_STROMA-segm_dice': VisConfigs.REGION_COLORS['STROMA'],
        # 'hpf-nuclei_sTIL-segm_dice': VisConfigs.NUCLEUS_COLORS['sTIL'],
        'hpf-nuclei_TILsCell-segm_dice': VisConfigs.NUCLEUS_COLORS['TILsCell'],
        'roi-CTA-score_rmse': VisConfigs.REGION_COLORS['TILS'],
        # 'hpf-CTA-score_rmse': VisConfigs.NUCLEUS_COLORS['sTIL'],
        'hpf-CTA-score_rmse': VisConfigs.NUCLEUS_COLORS['TILsCell'],
        'roi-CTA-score_pearson_r': VisConfigs.REGION_COLORS['TILS'],
        # 'hpf-CTA-score_pearson_r': VisConfigs.NUCLEUS_COLORS['sTIL'],
        'hpf-CTA-score_pearson_r': VisConfigs.NUCLEUS_COLORS['TILsCell'],
    }
    mcolors = {k: [j / 255. for j in v] for k, v in mcolors.items()}

    # ROI
    ax = axes[0]
    ax = _plot_ax(ax=ax, col='roi-regions_TUMOR-segm_dice',
                  label='Tumor region')
    ax = _plot_ax(ax=ax, col='roi-regions_STROMA-segm_dice',
                  label='Stroma region')
    ax = _plot_ax(ax=ax, col='roi-regions_TILS-segm_dice', label='TILs region')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('DICE', fontsize=11)
    ax.set_title('ROI segm. accuracy', fontsize=14, fontweight='bold')
    ax.legend()

    # HPF
    ax = axes[1]
    # ax = _plot_ax(ax=ax, col='hpf-regions_TUMOR-segm_dice', label='Tumor region')
    # ax = _plot_ax(ax=ax, col='hpf-regions_STROMA-segm_dice', label='Stroma region')
    # ax = _plot_ax(ax=ax, col='hpf-nuclei_sTIL-segm_dice', label='TILs nuclei')
    ax = _plot_ax(ax=ax, col='hpf-nuclei_TILsCell-segm_dice',
                  label='TILs nuclei')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('DICE', fontsize=11)
    ax.set_title('HPF segm. accuracy', fontsize=14, fontweight='bold')
    ax.legend()

    # Composite sTILs score RMSE
    ax = axes[2]
    ax = _plot_ax(ax=ax, col='roi-CTA-score_rmse', label='ROI CTA score',
                  fill=False)
    ax = _plot_ax(ax=ax, col='hpf-CTA-score_rmse', label='HPF CTA score',
                  fill=False)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('CTA score error', fontsize=14, fontweight='bold')
    ax.legend()

    # Composite sTILs score correlation
    ax = axes[3]
    ax = _plot_ax(ax=ax, col='roi-CTA-score_pearson_r', label='ROI CTA score',
                  fill=False)
    ax = _plot_ax(ax=ax, col='hpf-CTA-score_pearson_r', label='HPF CTA score',
                  fill=False)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Pearson R', fontsize=11)
    ax.set_title('CTA score correl.', fontsize=14, fontweight='bold')
    ax.legend()

    # now save
    plt.tight_layout(pad=0.4)
    plt.savefig(savename + '.svg')
    plt.close()
