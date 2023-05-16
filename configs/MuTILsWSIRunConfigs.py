import os
from os.path import join as opj
from pandas import read_csv


HOME = os.path.expanduser('~')
BASEPATH = opj(HOME, 'Desktop', 'cTME')


class BaseConfigs:

    DEBUG = True  # IMPORTANT: debug??

    # COHORT = 'NiceExamplesForMuTILsPreprint'
    # COHORT = 'TCGA_BRCA'
    COHORT = 'CPSII_40X'
    # COHORT = 'CPS3_40X'
    # COHORT = 'plco_breast'

    N_SUBSETS = 1
    # N_SUBSETS = 8 if not DEBUG else 1
    # N_SUBSETS = 16 if not DEBUG else 1

    MODELNAME = 'mutils_06022021'


class GrandChallengeConfigs:

    WORKPATH = opj(HOME, 'Desktop', 'TILS_CHALLENGE')
    INPUT_PATH = opj(WORKPATH, '1_INPUT', BaseConfigs.COHORT)
    OUTPUT_PATH = opj(WORKPATH, '2_OUTPUT', BaseConfigs.COHORT)
    BASEMODELPATH = opj(WORKPATH, '0_MODELS', BaseConfigs.MODELNAME)

    restrict_to_vta = False

    save_wsi_mask = True
    save_annotations = True
    save_nuclei_meta = False
    save_nuclei_props = False

    grandch = True

    # uncomment below to run on GC platform which runs on one slide at a time
    # so the slides will overwrite these files
    gcpaths = {
        'roilocs_in': opj(INPUT_PATH, "regions-of-interest.json"),
        'cta2vta': opj(WORKPATH, '0_MODELS', 'Calibrations.json'),
        'roilocs_out': opj(OUTPUT_PATH, 'regions-of-interest.json'),
        'result_file': opj(OUTPUT_PATH, 'results.json'),
        'tilscore_file': opj(OUTPUT_PATH, 'til-score.json'),
        'detect_file': opj(OUTPUT_PATH, 'detected-lymphocytes.json'),
        'wsi_mask': opj(OUTPUT_PATH, 'images', 'segmented-stroma'),
    }
    # gcpaths = None  # each slide has its own folder (no overwrite)

    topk_rois = 300
    topk_rois_sampling_mode = "weighted"
    topk_salient_rois = 300
    vlres_scorer_kws = {
        'check_tissue': True,
        'tissue_percent': 25,
        'pixel_overlap': 0,
    }


class FullcTMEConfigs:

    # INPUT_PATH = opj('/data', 'MUTILS_INPUT', BaseConfigs.COHORT)
    # OUTPUT_PATH = opj('/data', 'MUTILS_OUTPUT', BaseConfigs.COHORT)
    INPUT_PATH = opj('/input', BaseConfigs.COHORT)
    OUTPUT_PATH = opj('/output', BaseConfigs.COHORT)
    BASEMODELPATH = opj(
        BASEPATH, 'results', 'mutils', 'models', BaseConfigs.MODELNAME
    )

    restrict_to_vta = False

    save_wsi_mask = True
    save_annotations = False
    save_nuclei_meta = True
    save_nuclei_props = True

    grandch = False
    gcpaths = None
    topk_rois = None
    topk_rois_sampling_mode = "stratified"
    topk_salient_rois = None
    vlres_scorer_kws = None


class NiceExamplesForMuTILsPreprintConfigs:

    INPUT_PATH = opj('/input', BaseConfigs.COHORT)
    OUTPUT_PATH = opj('/output', BaseConfigs.COHORT)
    BASEMODELPATH = opj(
        BASEPATH, 'results', 'mutils', 'models', BaseConfigs.MODELNAME
    )

    restrict_to_vta = False

    save_wsi_mask = True
    save_annotations = False
    save_nuclei_meta = False
    save_nuclei_props = False

    grandch = False
    gcpaths = None
    topk_rois = None
    topk_rois_sampling_mode = "stratified"
    topk_salient_rois = None
    vlres_scorer_kws = None


class RunConfigs:

    # IMPORTANT: THIS SWITCHES THE ANALYSIS TYPE
    # cfg = NiceExamplesForMuTILsPreprintConfigs
    # cfg = GrandChallengeConfigs
    cfg = FullcTMEConfigs

    os.makedirs(cfg.OUTPUT_PATH, exist_ok=True)

    # Configure logger. This must be done BEFORE importing histolab modules
    from MuTILs_Panoptic.utils.MiscRegionUtils import \
        get_configured_logger, load_region_configs

    LOGDIR = opj(cfg.OUTPUT_PATH, 'LOGS')
    os.makedirs(LOGDIR, exist_ok=True)
    LOGGER = get_configured_logger(
        logdir=LOGDIR, prefix='MuTILsWSIRunner', tofile=True
    )

    # model weights
    MODEL_PATHS = {}
    for f in range(1, 6):
        MODEL_PATHS[f'{BaseConfigs.MODELNAME}-fold{f}'] = opj(
            cfg.BASEMODELPATH, f'fold_{f}', f'{BaseConfigs.MODELNAME}_fold{f}.pt'
        )

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Subdividing cohort to run on multiple docker instances

    from MuTILs_Panoptic.utils.GeneralUtils import splitlist

    ALL_SLIDENAMES = os.listdir(cfg.INPUT_PATH)

    if BaseConfigs.COHORT.startswith('TCGA') and cfg.restrict_to_vta:
        # RESTRICT TO ROBERTO SALGADO ASSESSED SLIDES
        from MuTILs_Panoptic.utils.GeneralUtils import splitlist
        SLIDENAMES = read_csv(opj(
            BASEPATH, 'data', 'tcga-clinical',
            'PRIVATE_RSalgado_TCGA_TILScores.csv'
        )).iloc[:, 0].to_list()
        SLIDENAMES = [j[:12] for j in SLIDENAMES]
        SLIDENAMES = [j for j in ALL_SLIDENAMES if j[:12] in SLIDENAMES]

    elif BaseConfigs.COHORT.endswith('CPS2') and cfg.restrict_to_vta:
        # RESTRICT TO TED ASSESSED SLIDES (CPS2)
        acs_vta = read_csv(
            opj(BASEPATH, 'data', 'acs-clinical',
                'CPSII_BRCA_FacilityIDs_20210331.csv'),
            index_col=0)
        acs_vta.rename(columns={'TILS_STR': 'vta'}, inplace=True)
        acs_vta = acs_vta.loc[:, 'vta'].map(lambda x: float(x) / 100).dropna()
        SLIDENAMES = list(acs_vta.index)
        SLIDENAMES = [j for j in ALL_SLIDENAMES if j.split('_')[0] in SLIDENAMES]

    else:
        SLIDENAMES = ALL_SLIDENAMES
        SLIDENAMES.sort()
        SLIDENAMES = splitlist(
            SLIDENAMES, len(SLIDENAMES) // BaseConfigs.N_SUBSETS
        )

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    RUN_KWARGS = dict(

        # paths & slides
        model_configs=load_region_configs(
            opj(cfg.BASEMODELPATH, 'region_model_configs.py'), warn=False
        ),
        model_paths=MODEL_PATHS,
        slides_path=cfg.INPUT_PATH,
        base_savedir=cfg.OUTPUT_PATH,

        # size params
        # roi_side_hres=512 if BaseConfigs.DEBUG else 1024,
        roi_side_hres=1024,
        discard_edge_hres=0,  # keep 0 -> slow + can't get the gap to be exact
        logger=LOGGER,

        # Defined in cfg
        save_wsi_mask=cfg.save_wsi_mask,
        save_annotations=cfg.save_annotations,
        save_nuclei_meta=cfg.save_nuclei_meta,
        save_nuclei_props=cfg.save_nuclei_props,
        grandch=cfg.grandch,
        gcpaths=cfg.gcpaths,
        topk_rois=cfg.topk_rois,
        topk_rois_sampling_mode=cfg.topk_rois_sampling_mode,
        topk_salient_rois=cfg.topk_salient_rois,
        vlres_scorer_kws=cfg.vlres_scorer_kws,

        _debug=BaseConfigs.DEBUG,
    )
