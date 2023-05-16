import os
from os.path import join as opj
from glob import glob
from pandas import read_csv, DataFrame, concat
import numpy as np
import shutil
import matplotlib.pylab as plt
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.special import softmax
from imageio import imwrite
import torch
import argparse

from MuTILs_Panoptic.utils.GeneralUtils import (
    AllocateGPU, reverse_dict, calculate_mcc
)

parser = argparse.ArgumentParser(description='MuTILs analytical validation.')
parser.add_argument('-g', type=int, default=[0], nargs='+', help='gpu(s)')
parser.add_argument(
    '--bst', type=int, default=0, help='bootstrapped (1) or manual truth (0)?'
)
parser.add_argument(
    '--tcga', type=int, default=1, help='TCGA (1) or CPS2 (0)?'
)
CLIARGS = parser.parse_args()
CLIARGS.bst = bool(CLIARGS.bst)
CLIARGS.tcga = bool(CLIARGS.tcga)

# GPU allocation MUST happen before importing other modules
AllocateGPU(GPUs_to_use=CLIARGS.g)

from MuTILs_Panoptic.mutils_panoptic.MuTILsInference import \
    MutilsInferenceRunner
from MuTILs_Panoptic.utils.GeneralUtils import calculate_4x4_statistics
from MuTILs_Panoptic.utils.TorchUtils import t2np
from MuTILs_Panoptic.mutils_panoptic.RegionDatasetLoaders import MuTILsDataset
from MuTILs_Panoptic.utils.MiscRegionUtils import (
    load_region_configs, _aggregate_semsegm_stats,
)
from MuTILs_Panoptic.mutils_panoptic.MuTILs import MuTILsEvaluator
from MuTILs_Panoptic.utils.MiscRegionUtils import \
    map_bboxes_using_hungarian_algorithm
from MuTILs_Panoptic.utils.RegionPlottingUtils import \
    get_visualization_ready_combined_mask


class MutilsAnalyticValidator(MutilsInferenceRunner):
    """This is an extension of the MuTILsEvaluator() class.

    The MuTILsEvaluator() class is a lightweight evaluation class to monitor
    the progess of the training process though calculation of semantic segment.
    accuracy. However, for the final paper-ready validation we need to do a
    more extensive "proper" validation, including:

    - Pixel confusion matrix for regions
    - Improving nuclear predictions by using watershed and pixel major. voting
    - Extracting and saving nuclear morphometric features for DTALE
    - Save soft probabilities &/or deep nuclear features
    - etc ...

    """

    def __init__(
            self,
            model_path: dict,
            model_configs,
            savepath: str,
            dataset: MuTILsDataset,
            *,
            aggregate_region_superclasses=False,
            roi_acceptable_region_misclassifs=None,
            no_watershed_nucleus_classes=None,
            maskout_regions_for_cnorm=None,
            min_nucl_size=5,
            max_nucl_size=90,
            extract_nprops=False,
            nprops_kwargs=None,
            vis_freq=10,
            _fold=1,
            _debug=False,
            _remove_existing=True,
    ):
        """"""
        super().__init__(
            model_configs=model_configs,
            model_path=model_path,
            roi_side_hres=dataset.roi_cropper.size[0],
            discard_edge_hres=0,
            cnorm=False,  # dataset already color normalized
            no_watershed_nucleus_classes=no_watershed_nucleus_classes,
            maskout_regions_for_cnorm=maskout_regions_for_cnorm,
            min_nucl_size=min_nucl_size,
            max_nucl_size=max_nucl_size,
            nprops_kwargs=nprops_kwargs,
            _debug=_debug,
        )

        self.savepath = savepath
        self.dataset = dataset
        self.aggregate_region_superclasses = aggregate_region_superclasses
        self.extract_nprops = extract_nprops
        self._fold = _fold
        self.vis_freq = vis_freq

        # acceptable roi misclassifs should be codes not names
        if roi_acceptable_region_misclassifs is None:
            self.roi_acceptable_region_misclassifs = None
        else:
            self.roi_acceptable_region_misclassifs = {
                roin: [
                    (self.rcd[pair[0]], self.rcd[pair[1]])
                    if pair[1] is not None
                    else (self.rcd[pair[0]], None)
                    for pair in acceptable
                ]
                for roin, acceptable in
                roi_acceptable_region_misclassifs.items()
            }

        # some sanity checks
        assert self.dataset.roi_cropper.iscentral, "No random crop!"
        assert self.dataset.roi_cropper.plusminus is None, "No random crop!"

        # some prepwork
        self._create_directories(_remove_existing)
        self.slide_roidxs = self.dataset.get_slide_roidxs()
        self.evaluator = MuTILsEvaluator()

        # params used later
        self._slmonitor = None
        self._sldname = None
        self._roidx = None
        self._roiname = None
        self._rmonitor = None
        self._rmonitor = None
        self._vis_flag = None

    def run(self):
        """
        Predict all slides and get accuracy metrics
        """
        # iterate through slides
        nslides = len(self.dataset.slides)
        for slno, self._sldname in enumerate(self.dataset.slides):
            self._slmonitor = (
                f"fold {self._fold}: "
                f"slide {slno + 1} of {nslides} ({self._sldname})"
            )
            print(self._slmonitor)

            if self._debug and slno > 1:
                break

            self.run_slide()

        # aggregate stats from all slides
        self._sldname = 'OVERALL'
        self._save_slide_stats(
            nuclei_meta=self._read_and_concat_all_nuclei_metas()
        )
        self.parse_nice_stats_tables()

    def parse_nice_stats_tables(self):
        """
        Parse nice stats tables for paper
        """
        self._nice_segmentation_tables()
        self._nice_detection_tables()
        self._nice_classification_tables()
        self._nice_confusion_tables()

    def run_slide(self):
        """
        Run pipeline for all rois in a single slide then aggregate.
        """
        roidxs = self.slide_roidxs[self._sldname]

        # iterate through rois in this slide
        for rno, self._roidx in enumerate(roidxs):
            self._roiname = self.dataset.roinames[self._roidx]

            self._rmonitor = f"{self._slmonitor}: roi {rno + 1} of {len(roidxs)}"
            print(self._rmonitor)
            self._vis_flag = rno % self.vis_freq == 0

            if self._debug and rno > 1:
                break

            self.run_roi()

        # calculate stats for this slide
        self._save_slide_stats(
            nuclei_meta=read_csv(
                opj(self.savepath, 'nucleiMeta', f'{self._sldname}.csv')),
        )

    def run_roi(self):
        """
        Run pipeline for single roi.
        """
        # do inference and save segmentation stats
        rgb, preds = self._predict_roi()

        if preds is None:
            print(f"{self._rmonitor}: SKIPPING ROI (NO NUCLEI!) ...")
            return

        try:
            # match nuclei from preds to truth & save
            preds['classif_df'] = self._append_true_nuclei_df(
                preds['classif_df'])
            where = opj(self.savepath, 'nucleiMeta', f'{self._sldname}.csv')
            preds['classif_df'].to_csv(
                where, header=not os.path.exists(where), mode='a')

            if not self.extract_nprops:
                return

            # extract and save nuclei props
            nprops = self.get_nuclei_props_df(rgb=rgb, preds=preds)
            where = opj(self.savepath, 'nucleiProps', f'{self._sldname}.csv')
            nprops.to_csv(where, header=not os.path.exists(where), mode='a')

        except Exception as e:
            print('\n')
            print(e.__repr__())
            print('\n')

    @torch.no_grad()
    def do_inference(self, roi_data):
        """Do MuTILs model inference."""
        batchdata = [
            {k: v.to(self.device) for k, v in bd.items()}
            for bd in roi_data]
        inference = self.model(batchdata)
        return inference

    # HELPERS -----------------------------------------------------------------

    def _nice_confusion_tables(self):
        """Parse nice segmentation tables"""
        for ispre in [False, True]:
            for issuper in [False, True]:
                self._nice_conf(issuper, ispre)

    def _nice_conf(self, issuper, ispre):
        fname = (
            f"{'Unconstrained.' if ispre else ''}"
            "ConfMatrices."
            f"{'SuperClass' if issuper else 'StandardClass'}"
        )
        stats = read_csv(opj(self.savepath, f'{fname}.csv'))
        stats = stats.loc[stats.loc[:, 'slide'] == 'OVERALL', :].sum(0)
        stats.index = [j.replace('Confusion.', '') for j in stats.index]
        # init confusion matrix
        if issuper:
            clsnames = list(self.rcc.SUPERNUCLEUS_CODES.keys())
        else:
            clsnames = list(self.rcc.NUCLEUS_CODES.keys())
        clsnames.remove('EXCLUDE')
        clsnames.remove('BACKGROUND')
        conf = DataFrame(0., index=clsnames, columns=clsnames)
        for tc in clsnames:
            for pc in clsnames:
                conf.loc[tc, pc] = stats[f'trueClass-{tc}_predictedClass-{pc}']
        nice = fname.replace('ConfMatrices', 'ConfusionMatrix')
        conf.to_csv(opj(self.savepath, 'niceTables', f'{nice}.csv'))

    def _nice_classification_tables(self):
        """Parse nice segmentation tables"""
        for ispre in [False, True]:
            for issuper in [False, True]:
                self._nice_classif(issuper, ispre)

    def _nice_classif(self, issuper, ispre):
        fname = (
            f"{'Unconstrained.' if ispre else ''}"
            "ClassifStats."
            f"{'SuperClass' if issuper else 'StandardClass'}"
        )
        stats = read_csv(opj(self.savepath, f'{fname}.csv'))
        stats = stats.loc[stats.loc[:, 'slide'] == 'OVERALL', :]
        stats.columns = [j.replace('Classif.', '') for j in stats.columns]
        stats = stats.rename(columns={
            'Overall.MicroAUROC': 'Overall.AUROC',
            'Overall.total': 'Overall.Ntrue',
        })
        # which columns to keep and what to name them
        clsnames = ['Overall']
        if issuper:
            clsnames += list(self.rcc.SUPERNUCLEUS_CODES.keys())
        else:
            clsnames += list(self.rcc.NUCLEUS_CODES.keys())
        clsnames.remove('EXCLUDE')
        clsnames.remove('BACKGROUND')
        table_cols = [' ', 'Fold'] + clsnames
        table = []
        # assign relevant metrics
        metrics = {
            'Ntrue': 'N',
            'accuracy': 'Accuracy',
            'MCC': 'MCC',
            'AUROC': 'AUROC',
            'MacroAUROC': 'MacroAUROC',
        }
        for met, nicemet in metrics.items():
            # title row
            row = DataFrame(columns=table_cols, index=[0])
            row.loc[0, ' '] = nicemet
            table.append(row)
            # results for each fold
            cols = [j for j in stats.columns if j.endswith(f'.{met}')]
            df = stats.loc[:, ['fold'] + cols].copy()
            # make as percentage & round
            if nicemet != 'N':
                df.loc[:, cols] = np.round(100 * df.loc[:, cols], 1)
            df.columns = ['Fold'] + [j.split('.')[0] for j in cols]
            table.append(df)
        # concat and save table
        table = concat(table, 0)
        nice = fname.replace('ClassifStats', 'Classification')
        table.to_csv(
            opj(self.savepath, 'niceTables', f'{nice}.csv'), index=False)

    def _nice_detection_tables(self):
        """Parse nice detection table"""
        stats_df = read_csv(opj(self.savepath, f'DetStats.csv'))
        stats_df = stats_df.loc[stats_df.loc[:, 'slide'] == 'OVERALL', :]
        mapped = {
            'fold': 'Fold',
            'Detection.total': 'N',
            'Detection.AP@.5': 'AP@.5',
            'Detection.F1': 'F1 score',
            'Detection.precision': 'Precision',
            'Detection.recall': 'Recall',
        }
        table = stats_df.loc[:, mapped.keys()]
        table.columns = mapped.values()
        table.iloc[:, 2:] = np.round(100 * table.iloc[:, 2:], 1)
        table.to_csv(
            opj(self.savepath, 'niceTables', f'Detection.csv'),
            index=False)

    def _nice_segmentation_tables(self):
        """Parse nice segmentation tables"""
        stats_df = read_csv(opj(self.savepath, f'SegmStats.csv'))
        stats_df = stats_df.loc[stats_df.loc[:, 'slide'] == 'OVERALL', :]
        self._nice_segm(stats_df, isreg=True, ispre=False)
        self._nice_segm(stats_df, isreg=False, ispre=False)
        self._nice_segm(stats_df, isreg=False, ispre=True)

    def _nice_segm(self, stats_df, isreg=True, ispre=False):
        """"""
        # regions vs nuclei vs unconstrained nuclei
        pfx = 'Segmentation.'
        if isreg:
            pfx += 'roi-regions'
        elif ispre:
            pfx += 'hpf-prenuclei'
        else:
            pfx += 'hpf-nuclei'
        # which columns to keep and what to name them
        clsnames = ['OVERALL']
        if isreg:
            clsnames += list(self.rcc.REGION_CODES.keys())
        else:
            clsnames += list(self.rcc.NUCLEUS_CODES.keys())
        if 'EXCLUDE' in clsnames:
            clsnames.remove('EXCLUDE')
        table_cols = [' ', 'Fold'] + clsnames
        table = []
        # assign relevant metrics
        metrics = {
            'pixel_accuracy': 'Pixel Accuracy',
            'segm_dice': 'DICE',
            'segm_iou': 'IOU'
        }
        for met, nicemet in metrics.items():
            # title row
            row = DataFrame(columns=table_cols, index=[0])
            row.loc[0, ' '] = nicemet
            table.append(row)
            # results for each fold
            cols = [
                j for j in stats_df.columns
                if j.startswith(f'{pfx}_') and j.endswith(f'-{met}')]
            df = stats_df.loc[:, ['fold'] + cols].copy()
            # make as percentage & round
            df.loc[:, cols] = np.round(100 * df.loc[:, cols], 1)
            df.columns = ['Fold'] + [
                j.replace(f'{pfx}_', '').replace(f'-{met}', '') for j in cols]
            table.append(df)
        # concat and save table
        table = concat(table, 0)
        if isreg:
            table.columns = [j.title() for j in table.columns]
        nicepfx = pfx.replace('roi-', '').replace('hpf-', '').title()
        table.to_csv(
            opj(self.savepath, 'niceTables', f'{nicepfx}.csv'), index=False)

    def _read_and_concat_all_nuclei_metas(self):
        """Concat all nuclei metas from all slides."""
        dfs = []
        for sld in self.dataset.slides:
            try:
                dfs.append(read_csv(
                    opj(self.savepath, 'nucleiMeta', f'{sld}.csv'), index_col=0
                ))
            except FileNotFoundError:
                pass
        return concat(dfs, axis=0, ignore_index=True)

    def _save_slide_stats(self, nuclei_meta):
        """Aggregate segmentation, detection and classification stats."""
        self._save_slide_segment_stats()
        self._save_slide_detection_stats(nuclei_meta)
        for constrained in [True, False]:
            for issuper in [True, False]:
                self._save_slide_classif_stats(
                    nuclei_meta, isconstr=constrained, issuper=issuper)

    def _save_slide_classif_stats(self, df0, isconstr: bool, issuper: bool):
        """
        Save nucleus classification stats and confusions for slide.
        """
        # prepwork
        df = df0.loc[df0.loc[:, 'Matching'] == 'Matched', :]
        df = df.loc[df.loc[:, 'Truth.RawClass'] != 'unlabeled', :]
        pfx = '' if isconstr else 'Unconstrained.'
        pfx2 = 'SuperClassif' if issuper else 'Classif'
        sfx = 'SuperClass' if issuper else 'StandardClass'
        clmap = self.rcc.SUPERNUCLEUS_CODES if issuper else self.rcc.NUCLEUS_CODES
        rclmap = reverse_dict(clmap)
        clset = set(clmap.keys())
        clset = clset.difference({'EXCLUDE', 'BACKGROUND'})
        pred = df.loc[:, f'{pfx}Classif.{sfx}'].map(clmap).values
        true = df.loc[:, f'Truth.{sfx}'].map(clmap).values

        # get and save flat confusion matrix
        self._save_slide_confusion_matrix(
            clset=clset, clmap=clmap, true=true, pred=pred, pfx=pfx, sfx=sfx)

        # init accuracy dict with overall accuracy
        n_total = len(true)
        clmetrics = {
            'fold': self._fold,
            'slide': self._sldname,
            'Classif.Overall.total': n_total,
            'Classif.Overall.accuracy': np.mean(0 + (pred == true)),
            'Classif.Overall.MCC': calculate_mcc(true, pred),
            'Classif.Overall.MicroAUROC': np.nan,
            'Classif.Overall.MacroAUROC': np.nan,
        }
        for cls_name in clset:
            for metric in ['Ntrue', 'accuracy', 'MCC', 'AUROC']:
                clmetrics[f'Classif.{cls_name}.{metric}'] = np.nan

        # class-by-class accuracy
        unique_classes = np.unique(true).tolist()
        n_classes = len(unique_classes)
        trg = np.zeros((n_total, n_classes))
        scr = np.zeros((n_total, n_classes))
        for cid, cls in enumerate(unique_classes):
            cls_name = rclmap[cls]
            # Accuracy
            tr = 0 + (true == cls)
            pr = 0 + (pred == cls)
            clmetrics[f'Classif.{cls_name}.Ntrue'] = np.sum(tr)
            clmetrics[f'Classif.{cls_name}.accuracy'] = np.mean(0 + (tr == pr))
            # Mathiew's Correlation Coefficient
            clmetrics[f'Classif.{cls_name}.MCC'] = calculate_mcc(tr, pr)
            # ROC AUC
            if n_classes > 1:
                trg[:, cid] = tr
                scr[:, cid] = df.loc[:, f'{pfx}{pfx2}Probab.{cls_name}'].values
                clmetrics[f'Classif.{cls_name}.AUROC'] = roc_auc_score(
                    y_true=trg[:, cid], y_score=scr[:, cid])

        # renormalize with softmax & get overall AUROC
        if n_classes > 1:
            scr = softmax(scr, -1)
            clmetrics['Classif.Overall.MicroAUROC'] = roc_auc_score(
                y_true=trg, y_score=scr, multi_class='ovr', average='micro')
            clmetrics['Classif.Overall.MacroAUROC'] = roc_auc_score(
                y_true=trg, y_score=scr, multi_class='ovr', average='macro')

        # now save
        where = opj(self.savepath, f'{pfx}ClassifStats.{sfx}.csv')
        statsdf = DataFrame.from_records([clmetrics])
        statsdf.to_csv(where, header=not os.path.exists(where), mode='a')

    def _save_slide_confusion_matrix(self, clset, clmap, true, pred, pfx, sfx):
        """Save flattened confusion matrix for slide"""
        confmat = {'fold': self._fold, 'slide': self._sldname}
        for tc in clset:
            for pc in clset:
                coln = f'Confusion.trueClass-{tc}_predictedClass-{pc}'
                tcc, pcc = clmap[tc], clmap[pc]
                keep1 = 0 + (true == tcc)
                keep2 = 0 + (pred == pcc)
                confmat[coln] = np.sum(0 + ((keep1 + keep2) == 2))
        # now save
        where = opj(self.savepath, f'{pfx}ConfMatrices.{sfx}.csv')
        statsdf = DataFrame.from_records([confmat])
        statsdf.to_csv(where, header=not os.path.exists(where), mode='a')

    def _save_slide_detection_stats(self, df):
        """Save nucleus detection stats for slide."""
        # get average precision stats
        y_true = np.zeros(df.shape[0])
        y_true[df.loc[:, 'Matching'] == 'Matched'] = 1
        y_true[df.loc[:, 'Matching'] == 'UnmatchedTruth'] = 1
        y_score = np.array(1. - df.loc[:, 'ClassifProbab.BACKGROUND'])
        y_score[df.loc[:, 'Matching'] == 'UnmatchedTruth'] = 0.
        keep = np.argwhere(np.isfinite(y_score))
        y_score = y_score[keep]
        y_true = y_true[keep]
        stats = {
            'fold': self._fold,
            'slide': self._sldname,
            'Detection.AP@.5': average_precision_score(y_true, y_score),
            'Detection.Random': np.mean(y_true),
        }
        # append discretized stats
        simple = calculate_4x4_statistics(
            TP=np.sum(0 + (df.loc[:, 'Matching'] == 'Matched')),
            FP=np.sum(0 + (df.loc[:, 'Matching'] == 'UnmatchedPred')),
            FN=np.sum(0 + (df.loc[:, 'Matching'] == 'UnmatchedTruth')),
            add_eps_to_tn=False,
        )
        stats.update({f'Detection.{k}': v for k, v in simple.items()})
        # now save
        where = opj(self.savepath, f'DetStats.csv')
        statsdf = DataFrame.from_records([stats])
        statsdf.to_csv(where, header=not os.path.exists(where), mode='a')

    def _save_slide_segment_stats(self):
        """Aggregate semantic segm. stats from rois beloning to same slide."""
        # aggregate rois within slide, or all slides in dataset?
        if self._sldname == 'OVERALL':
            dfpath = opj(self.savepath, 'SegmStats.csv')
            colpfx = 'Segmentation.'
        else:
            dfpath = opj(self.savepath, 'segmStats', f'{self._sldname}.csv')
            colpfx = ''
        # aggregate stats for slide & region superclasses

        do_not_aggregate = (
                (self._sldname == 'OVERALL') or
                (not self.aggregate_region_superclasses)
        )
        segmstats = _aggregate_semsegm_stats(
            read_csv(dfpath),
            colpfx=colpfx,
            supermap=None if do_not_aggregate else {
                'NORMAL': 'TUMOR',
                'JUNK': 'OTHER',
                'BLOOD': 'OTHER',
                'WHITE': 'OTHER',
            },
        )
        stats = {'fold': self._fold, 'slide': self._sldname}
        stats.update({f'Segmentation.{k}': v for k, v in segmstats.items()})
        # now save
        where = opj(self.savepath, f'SegmStats.csv')
        statsdf = DataFrame.from_records([stats])
        statsdf.to_csv(where, header=not os.path.exists(where), mode='a')

    def _append_true_nuclei_df(self, pred_df):
        """
        Match nuclei from prediction & truth and get combined dataframe.
        """
        truth_df = self._get_true_nuclei_df()
        # match using linear sum assignment with bbox IOU thresh. of 0.5
        matched_truth, matched_pred, unmatched_truth, unmatched_pred = \
            map_bboxes_using_hungarian_algorithm(
                bboxes1=np.int32(truth_df.loc[:, [
                                                     'Truth.Xmin',
                                                     'Truth.Ymin',
                                                     'Truth.Xmax',
                                                     'Truth.Ymax']].values),
                bboxes2=np.int32(pred_df.loc[:, [
                                                    'Identifier.Xmin',
                                                    'Identifier.Ymin',
                                                    'Identifier.Xmax',
                                                    'Identifier.Ymax']].values),
                min_iou=0.5,
            )
        # concat matched nuclei
        matched_truth_df = truth_df.iloc[matched_truth.tolist(), :].copy()
        matched_pred_df = pred_df.iloc[matched_pred.tolist(), :].copy()
        matched_truth_df.reset_index(inplace=True, drop=True)
        matched_pred_df.reset_index(inplace=True, drop=True)
        matched_df = concat([matched_truth_df, matched_pred_df], axis=1)
        matched_df.insert(0, 'Matching', 'Matched')
        # append unmatched nuclei
        unmatched_truth_df = truth_df.iloc[unmatched_truth.tolist(), :]
        unmatched_truth_df.insert(0, 'Matching', 'UnmatchedTruth')
        unmatched_pred_df = pred_df.iloc[unmatched_pred.tolist(), :]
        unmatched_pred_df.insert(0, 'Matching', 'UnmatchedPred')
        classif_df = concat(
            [matched_df, unmatched_truth_df, unmatched_pred_df],
            axis=0, ignore_index=True)
        classif_df.insert(0, 'roiname', self._roiname)
        classif_df.insert(0, 'slide', self._sldname)
        classif_df.insert(0, 'fold', self._fold)

        return classif_df

    def _get_true_nuclei_df(self):
        """
        Get roi nuclei truth dataframe.
        """
        # read previously extracted nuclei locations
        istcga = self._roiname.startswith('TCGA')
        root = self.dataset.tcga_root if istcga else self.dataset.acs_root
        truthdf = read_csv(
            opj(root, 'csv', self._roiname.replace('.png', '.csv')),
            index_col=0)
        keepcols = [
            'left', 'top', 'right', 'bottom', 'type', 'raw_group', 'group']
        truthdf = truthdf.loc[:, keepcols]
        #
        # IMPORTANT NOTE:
        #  the ground truth boxes were extracted relative to the image at
        #  ORIGINAL mpp, while the cropper we are using here is at HPF mpp
        #
        # remove boxes outside central crop zone (all at original mpp)
        sf = self.dataset.original_mpp / self.dataset.hpf_mpp
        crop = self.dataset.roi_cropper.size[0] / sf  # initialized
        margin = (self.dataset.original_side - crop) // 2
        truthdf = truthdf.loc[truthdf.loc[:, 'left'] > margin, :]
        truthdf = truthdf.loc[truthdf.loc[:, 'top'] > margin, :]
        truthdf = truthdf.loc[truthdf.loc[:, 'right'] < crop + margin, :]
        truthdf = truthdf.loc[truthdf.loc[:, 'bottom'] < crop + margin, :]
        # adjust box coordinates given crop
        for corner in ['left', 'right', 'top', 'bottom']:
            truthdf.loc[:, corner] -= margin
        # map from original to hpf mpp
        truthdf.loc[:, ['left', 'top', 'right', 'bottom']] *= sf
        # rename columns to same notation as pred df
        truthdf = truthdf.rename(columns={
            'left': 'Truth.Xmin', 'top': 'Truth.Ymin',
            'right': 'Truth.Xmax', 'bottom': 'Truth.Ymax',
            'type': 'Truth.AnnotType',
            'raw_group': 'Truth.RawClass',
            'group': 'Truth.StandardClass',
        })
        truthdf.loc[:, 'Truth.SuperClass'] = \
            truthdf.loc[:, 'Truth.StandardClass'].map(
                self.rcc.NClass2Superclass_names)

        return truthdf

    def _predict_roi(self):
        """do inference and save semantic segmentation stats."""
        # do inference and save segmentation stats
        roi_data, roi_truth = self.dataset.__getitem__(self._roidx)
        inference = self.do_inference(roi_data=[roi_data])
        self._save_roi_segment_stats(inference, truth=[roi_truth])
        # refactor predictions & append truth
        roi_preds = self.refactor_inference(
            inference=inference,
            hres_ignore=roi_truth['highres_mask'][0, 0, ...] == 0)

        if roi_preds is None:
            return None, None

        # save mask
        imwrite(
            opj(self.savepath, 'combinedMasks', self._roiname),
            roi_preds['combined_mask'])

        # maybe save visualization for paper
        roi_rgb = np.moveaxis(t2np(roi_data['highres_rgb'][0, ...]), 0, -1)
        roi_rgb = np.uint8(roi_rgb * 255)
        if self._vis_flag:
            self._save_roi_visualization(
                rgb=roi_rgb, preds=roi_preds, truth=roi_truth)

        return roi_rgb, roi_preds

    def _save_roi_segment_stats(self, inference, truth):
        """Save raw semantic segmentation stats"""
        ignore = ['idx', 'roiname']
        truth = [{
            k: v.to(self.device) if k not in ignore else v for k, v in
            bd.items()}  # noqa
            for bd in truth]
        sseg_stats = self.evaluator(
            inference=inference,
            truth=truth,
            acceptable_region_misclassif=(
                self.roi_acceptable_region_misclassifs[self._roiname]
                if self.roi_acceptable_region_misclassifs is not None
                else [(self.rcd['STROMA'], self.rcd['TILS'])]
            ),
        )[0]
        del sseg_stats['epoch']
        sseg_stats.update({
            'roiname': self._roiname,
            'slide': self._sldname,
            'fold': self._fold,
        })
        sseg_stats = DataFrame.from_records([sseg_stats])
        where = opj(self.savepath, 'segmStats', f'{self._sldname}.csv')
        sseg_stats.to_csv(where, header=not os.path.exists(where), mode='a')

    def _save_roi_visualization(self, rgb, preds, truth):
        """Visualize to use for paper etc."""
        # prepare
        arrs = [rgb] + self._prep_for_vis(preds=preds, truth=truth)
        # now visualize
        fheight = 7
        fwidth = 28
        fig, ax = plt.subplots(1, 4, figsize=(fwidth, fheight), sharey=True)
        axno = -1
        for axis, arr in zip(ax, arrs):
            axno += 1
            kws = {} if axno == 0 else {
                'cmap': self.cfg.VisConfigs.COMBINED_CMAP,
                'interpolation': 'nearest',
            }
            axis.imshow(arr, **kws)
            axis.axis('off')
        fig.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        plt.savefig(opj(self.savepath, 'combinedVis', self._roiname))
        plt.close()

    @staticmethod
    def _prep_for_vis(preds, truth):
        """Prepare visualization ready np arrays"""
        gvm = get_visualization_ready_combined_mask
        true_mask = gvm(np.moveaxis(
            t2np(truth['highres_mask'][0, ...]), 0, -1))
        pred_mask = gvm(preds['combined_mask'])
        prepred_mask = gvm(preds['precombined_mask'])

        return [true_mask, pred_mask, prepred_mask]

    def _create_directories(self, _remove_existing):
        os.makedirs(self.savepath, exist_ok=True)
        if _remove_existing:
            for junk in glob(opj(self.savepath, '*.csv')):
                os.remove(junk)
        for subdir in [
            'nucleiMeta', 'nucleiProps', 'segmStats',
            'combinedMasks', 'combinedVis', 'niceTables',
        ]:
            where = opj(self.savepath, subdir)
            if _remove_existing:
                shutil.rmtree(where, ignore_errors=True)
            os.makedirs(where, exist_ok=True)


# %% ==========================================================================


def main():
    from MuTILs_Panoptic.mutils_panoptic.RegionDatasetLoaders import \
        get_cv_fold_slides

    DEBUG = False  # IMPORTANT: debug mode?

    # paths
    BASEPATH = opj(os.path.expanduser('~'), 'Desktop', 'cTME')
    MODELNAME = 'mutils_06022021'
    BASEMODELPATH = opj(BASEPATH, 'results', 'mutils', 'models', MODELNAME)
    SAVEPATH = opj(
        BASEPATH, 'results', 'mutils', f'AnalyticalValidation_{MODELNAME}')
    os.makedirs(SAVEPATH, exist_ok=True)

    # configs
    model_configs = load_region_configs(
        configs_path=opj(BASEMODELPATH, 'region_model_configs.py'), warn=False)
    mtp = model_configs.MuTILsParams
    mtp.test_dataset_params.update({
        'training': False,
        'crop_iscentral': True,
        'scale_augm_ratio': None,
        '_shuf': False,  # nice slide order
    })
    if DEBUG:
        mtp.test_dataset_params['roi_side'] = int(
            mtp.test_dataset_params['roi_side'] / 2)

    for FOLD in range(1, 6):

        # data loading
        if not CLIARGS.bst:
            mtp.root = opj(BASEPATH, 'data',
                           'ManualNucleiManualRegions_06062021')
        train_slides, test_slides = get_cv_fold_slides(
            train_test_splits_path=opj(mtp.root, 'train_test_splits'),
            fold=FOLD)

        # Restrict to TCGA where we have (mostly) reliable manual regions
        if CLIARGS.tcga:
            test_slides = [j for j in test_slides if j.startswith('TCGA')]
        else:
            test_slides = [j for j in test_slides if not j.startswith('TCGA')]

        dataset = MuTILsDataset(
            root=mtp.root, slides=test_slides, **mtp.test_dataset_params)

        # init & run
        validator = MutilsAnalyticValidator(
            model_path=opj(BASEMODELPATH, f'fold_{FOLD}', f'{MODELNAME}.pt'),
            model_configs=model_configs,
            savepath=opj(
                SAVEPATH,
                os.path.basename(dataset.root),
                'TCGA' if CLIARGS.tcga else 'CPS2'
            ),
            dataset=dataset,
            aggregate_region_superclasses=False,
            roi_acceptable_region_misclassifs=None,
            extract_nprops=CLIARGS.bst and CLIARGS.tcga,
            min_nucl_size=5,
            max_nucl_size=90,
            # vis_freq=1 if (DEBUG or not CLIARGS.bst) else 10,
            vis_freq=1 if CLIARGS.tcga else 1000,
            _fold=FOLD,
            _debug=DEBUG,
            _remove_existing=(not DEBUG) and (FOLD < 2),
        )
        validator.run()


if __name__ == '__main__':
    main()
