from collections import OrderedDict
from pandas import read_csv
from matplotlib.colors import ListedColormap
from copy import deepcopy
import os
from os.path import join as opj


from MuTILs_Panoptic.utils.GeneralUtils import (
    ordered_vals_from_ordered_dict, reverse_dict
)
from MuTILs_Panoptic.configs.nucleus_style_defaults import NucleusCategories


CONFIGPATH = os.path.dirname(__file__)


# from torchvision
def collate_fn(batch):
    return tuple(zip(*batch))


class RegionCategories(object):

    # GT codes dict for parsing into label mask
    GTCODE_PATH = opj(CONFIGPATH, 'region_GTcodes.csv')
    gtcodes_df = read_csv(GTCODE_PATH)
    gtcodes_df.index = gtcodes_df.loc[:, 'group']
    gtcodes_dict = gtcodes_df.to_dict(orient='index')

    # raw categories codes
    raw_categs = list(gtcodes_df.index)
    raw_categs_codes = {rc: v['GT_code'] for rc, v in gtcodes_dict.items()}

    # map from raw categories to main categories to be learned
    raw_to_main_categmap = {
        rc: v['main_classes'] for rc, v in gtcodes_dict.items()}
    main_categs_codes = {
        v['main_classes']: v['main_codes'] for v in gtcodes_dict.values()}
    main_categs = ordered_vals_from_ordered_dict(raw_to_main_categmap)

    # map from raw categories to super categories
    raw_to_super_categmap = {
        rc: v['super_classes'] for rc, v in gtcodes_dict.items()}
    super_categs_codes = {
        v['super_classes']: v['super_codes'] for v in gtcodes_dict.values()}
    super_categs = ordered_vals_from_ordered_dict(raw_to_super_categmap)

    # direct dict mapping from raw categs to main category gt codes
    raw_to_main_categs_codes = {}
    raw_codes_to_main_codes = {}
    for k, v in raw_to_main_categmap.items():
        raw_to_main_categs_codes[k] = main_categs_codes[v]
        raw_codes_to_main_codes[raw_categs_codes[k]] = main_categs_codes[v]

    # direct dict mapping from raw categs to super category gt codes
    raw_to_super_categs_codes = {}
    raw_codes_to_super_codes = {}
    for k, v in raw_to_super_categmap.items():
        raw_to_super_categs_codes[k] = super_categs_codes[v]
        raw_codes_to_super_codes[raw_categs_codes[k]] = super_categs_codes[v]


def get_combined_codes(region_codes, nucleus_codes):
    """Codes for combined region-nucleus masks."""
    _rcodes = {f'REGION-{k}': v for k, v in region_codes.items()}
    mx = max(_rcodes.values())
    _ncodes = {f'NUCLEUS-{k}': v + mx for k, v in nucleus_codes.items()}
    del _rcodes['REGION-EXCLUDE'], _ncodes['NUCLEUS-EXCLUDE'], \
        _ncodes['NUCLEUS-BACKGROUND']
    combined_codes = {'EXCLUDE': 0}
    combined_codes.update(_rcodes)
    combined_codes.update(_ncodes)
    combined_codes['TISSUE'] = max(combined_codes.values()) + 1
    rcombined_codes = reverse_dict(combined_codes)
    return combined_codes, rcombined_codes


class RegionCellCombination(object):

    REGION_CODES = deepcopy(RegionCategories.super_categs_codes)
    RREGION_CODES = reverse_dict(REGION_CODES)

    NUCLEUS_CODES = OrderedDict({
        'EXCLUDE': 0,  # ignore
        'CancerEpithelium': 1,
        'StromalCellNOS': 2,
        'ActiveStromalCellNOS': 3,
        'TILsCell': 4,
        'ActiveTILsCell': 5,
        'NormalEpithelium': 6,
        'OtherCell': 7,
        'UnknownOrAmbiguousCell': 8,
        'BACKGROUND': 9,  # non-nuclear (cytoplasm etc)
    })
    RNUCLEUS_CODES = reverse_dict(NUCLEUS_CODES)

    COMBINED_CODES, RCOMBINED_CODES = get_combined_codes(
        region_codes=REGION_CODES, nucleus_codes=NUCLEUS_CODES)

    # >>> SUPERCLASSES FOR ACCURACY PURPOSES <<<
    SUPERNUCLEUS_CODES = OrderedDict({
        'EXCLUDE': 0,  # ignore
        'EpithelialSuperclass': 1,
        'StromalSuperclass': 2,
        'TILsSuperclass': 3,
        'OtherSuperclass': 4,
        'AmbiguousSuperclass': 5,
        'BACKGROUND': 6,  # non-nuclear (cytoplasm etc)
    })
    NClass2Superclass_names = {
        'CancerEpithelium': 'EpithelialSuperclass',
        'StromalCellNOS': 'StromalSuperclass',
        'ActiveStromalCellNOS': 'StromalSuperclass',
        'TILsCell': 'TILsSuperclass',
        'ActiveTILsCell': 'TILsSuperclass',
        'NormalEpithelium': 'EpithelialSuperclass',
        'OtherCell': 'OtherSuperclass',
        'UnknownOrAmbiguousCell': 'AmbiguousSuperclass',
        'BACKGROUND': 'BACKGROUND',
    }
    RSUPERNUCLEUS_CODES = reverse_dict(SUPERNUCLEUS_CODES)
    NClass2Superclass_codes = {}
    for k, v in NClass2Superclass_names.items():
        NClass2Superclass_codes[NUCLEUS_CODES[k]] = SUPERNUCLEUS_CODES[v]

    # -> -> FOR ** TRAINING ** MUTILS <- <-
    # mapping from NuCLS MAIN classes to bootstrap nuclei classes
    # this is used to map the BOOTSTRAPPED truth from NuCLS model inference
    # to the standardized class set
    NuCLSCodes = NucleusCategories.main_categs_codes
    NuCLS2Bootstrap_names = {
        'tumor_nonMitotic': 'CancerEpithelium',
        'tumor_mitotic': 'UnknownOrAmbiguousCell',  # unreliable NuCLS mitotic
        'nonTILnonMQ_stromal': 'StromalCellNOS',
        'macrophage': 'ActiveStromalCellNOS',
        'lymphocyte': 'TILsCell',
        'plasma_cell': 'ActiveTILsCell',
        'other_nucleus': 'OtherCell',
        'AMBIGUOUS': 'UnknownOrAmbiguousCell',
    }
    NuCLS2Bootstrap_codes = {}
    for k, v in NuCLS2Bootstrap_names.items():
        NuCLS2Bootstrap_codes[NuCLSCodes[k]] = NUCLEUS_CODES[v]

    # -> -> FOR ** VALIDATING ** MUTILS <- <-
    # mapping from NuCLS RAW classes, including fov, to standardized set
    # this is used to map the MANUAL truth to the standardized class set
    RawNuCLSCodes = {
        k: v['GT_code'] for k, v in NucleusCategories.gtcodes_dict.items()}
    RawNuCLS2Bootstrap_names = {
        'fov_basic': 'BACKGROUND',
        'tumor': 'CancerEpithelium',
        'fibroblast': 'StromalCellNOS',
        'lymphocyte': 'TILsCell',
        'plasma_cell': 'ActiveTILsCell',
        'macrophage': 'ActiveStromalCellNOS',
        'mitotic_figure': 'CancerEpithelium',  # reliable as being tumor
        'vascular_endothelium': 'StromalCellNOS',
        'myoepithelium': 'StromalCellNOS',
        'apoptotic_body': 'UnknownOrAmbiguousCell',
        'neutrophil': 'OtherCell',
        'ductal_epithelium': 'NormalEpithelium',
        'eosinophil': 'OtherCell',
        'unlabeled': 'UnknownOrAmbiguousCell',
    }
    RawNuCLS2Bootstrap_codes = {}
    for k, v in RawNuCLS2Bootstrap_names.items():
        RawNuCLS2Bootstrap_codes[RawNuCLSCodes[k]] = NUCLEUS_CODES[v]

    # During bootstrapping, if NuCLS predicts this nucleus class in that
    # region assume that the label is actually x
    forced_bootstrap = {
        ('STROMA', 'CancerEpithelium'): 'ActiveStromalCellNOS',
        ('TILS', 'CancerEpithelium'): 'ActiveStromalCellNOS',
    }

    # for specific nucleus super-class, constrain super-regions allowed
    nuclei_regions = {
        'CancerEpithelium': ['TUMOR'],
        'StromalCellNOS': ['STROMA', 'TILS'],
        'ActiveStromalCellNOS': ['STROMA', 'TILS'],
        'TILsCell': ['STROMA', 'TILS'],
        'ActiveTILsCell': ['STROMA', 'TILS'],
        'NormalEpithelium': ['NORMAL'],
        'OtherCell': ['OTHER'],
        'UnknownOrAmbiguousCell': ['TUMOR', 'STROMA', 'JUNK', 'BLOOD', 'OTHER'],
        'BACKGROUND': ['TUMOR', 'STROMA', 'TILS', 'NORMAL', 'JUNK', 'BLOOD', 'OTHER', 'WHITE'],  # noqa
    }
    nuclei_regions_codes = {}
    for ncat, sreg in nuclei_regions.items():
        nc = NUCLEUS_CODES[ncat]
        nuclei_regions_codes[nc] = []
        for rcat in sreg:
            nuclei_regions_codes[nc].append(REGION_CODES[rcat])

    N_RCLASSES = len(REGION_CODES) - 1  # no exclude
    N_NCLASSES = len(NUCLEUS_CODES) - 1  # no exclude

    # get list where True means allowed region for this nucleus
    allowed_regions = []
    for nc in range(1, N_NCLASSES + 1):
        ars = []
        for k in range(N_RCLASSES):
            ars.append(True if k + 1 in nuclei_regions_codes[nc] else False)
        allowed_regions.append(ars)

    # region-derived nucleus labels if NOT compatible with mapping above
    regions_derived_nuclei = {
        'TUMOR': 'CancerEpithelium',
        'STROMA': 'StromalCellNOS',
        'TILS': 'TILsCell',
        'NORMAL': 'NormalEpithelium',
        'OTHER': 'OtherCell',
        'JUNK': 'UnknownOrAmbiguousCell',
        'BLOOD': 'UnknownOrAmbiguousCell',
        'WHITE': 'BACKGROUND',
        'EXCLUDE': 'EXCLUDE',
    }
    regions_derived_nuclei_codes = {}
    for rcat, ncat in regions_derived_nuclei.items():
        regions_derived_nuclei_codes[REGION_CODES[rcat]] = NUCLEUS_CODES[ncat]


def get_combined_colors(region_colors, nucleus_colors):
    """Get combined region-cell colors"""
    combined_colors = {}
    for k, v in RegionCellCombination.COMBINED_CODES.items():
        _regn = k.split('REGION-')[-1]
        _nucl = k.split('NUCLEUS-')[-1]
        if k == 'EXCLUDE':
            combined_colors[k] = region_colors['EXCLUDE']
        elif _regn in region_colors:
            combined_colors[k] = region_colors[_regn]
        elif _nucl in nucleus_colors:
            combined_colors[k] = nucleus_colors[_nucl]
    combined_colors['TISSUE'] = [128, 128, 128]
    combined_cmap = ListedColormap(
        [[i / 255. for i in v] for v in combined_colors.values()])

    return combined_colors, combined_cmap


class VisConfigs(object):

    REGION_COLORS = OrderedDict({
        'EXCLUDE': [0, 0, 0],
        'TUMOR': [192, 56, 255],
        'STROMA': [224, 133, 255],
        'TILS': [128, 128, 255],
        'NORMAL': [192, 56, 255],  # ugly hack so it'd groped with TUMOR
        'JUNK': [239, 255, 138],
        'BLOOD': [255, 33, 33],
        'OTHER': [168, 168, 168],
        'WHITE': [219, 219, 219],
    })

    REGION_CMAP = ListedColormap(
        [[i / 255. for i in v] for v in REGION_COLORS.values()])

    NUCLEUS_COLORS = OrderedDict({
        'EXCLUDE': REGION_COLORS['EXCLUDE'],
        'CancerEpithelium': [132, 0, 194],
        'StromalCellNOS': [209, 77, 255],
        'ActiveStromalCellNOS': [209, 77, 255],  # hack to group with fibrobl
        'TILsCell': [0, 0, 255],
        'ActiveTILsCell': [0, 0, 255],  # ugly hack to group with tils
        'NormalEpithelium': [132, 0, 194],  # ugly hack to group with cancer
        'OtherCell': [150, 150, 150],
        'UnknownOrAmbiguousCell': [100, 100, 100],
        'BACKGROUND': [219, 219, 219],
    })
    SUPERNUCLEUS_COLORS = OrderedDict({
        'EXCLUDE': REGION_COLORS['EXCLUDE'],
        'EpithelialSuperclass': [132, 0, 194],
        'StromalSuperclass': [209, 77, 255],
        'TILsSuperclass': [0, 0, 255],
        'OtherSuperclass': [150, 150, 150],
        'AmbiguousSuperclass': [100, 100, 100],
        'BACKGROUND': [219, 219, 219],
    })
    NUCLEUS_CMAP = ListedColormap(
        [[i / 255. for i in v] for v in NUCLEUS_COLORS.values()])
    SUPERNUCLEUS_CMAP = ListedColormap(
        [[i / 255. for i in v] for v in SUPERNUCLEUS_COLORS.values()])

    ALT_NUCLEUS_COLORS = OrderedDict({
        'EXCLUDE': REGION_COLORS['EXCLUDE'],
        'CancerEpithelium': [255, 0, 0],
        'StromalCellNOS': [2, 189, 64],
        'ActiveStromalCellNOS': [179, 255, 0],
        'TILsCell': [0, 0, 255],
        'ActiveTILsCell': [0, 255, 255],
        'NormalEpithelium': [51, 102, 153],
        'OtherCell': [150, 150, 150],
        'UnknownOrAmbiguousCell': [100, 100, 100],
        'BACKGROUND': [219, 219, 219],
    })
    ALT_SUPERNUCLEUS_COLORS = OrderedDict({
        'EXCLUDE': REGION_COLORS['EXCLUDE'],
        'EpithelialSuperclass': [255, 0, 0],
        'StromalSuperclass': [2, 189, 64],
        'TILsSuperclass': [0, 0, 255],
        'OtherSuperclass': [150, 150, 150],
        'AmbiguousSuperclass': [100, 100, 100],
        'BACKGROUND': [219, 219, 219],
    })
    ALT_NUCLEUS_CMAP = ListedColormap(
        [[i / 255. for i in v] for v in ALT_NUCLEUS_COLORS.values()])
    ALT_SUPERNUCLEUS_CMAP = ListedColormap(
        [[i / 255. for i in v] for v in ALT_SUPERNUCLEUS_COLORS.values()])

    COMBINED_COLORS, COMBINED_CMAP = get_combined_colors(REGION_COLORS, NUCLEUS_COLORS)
    SUPERCOMBINED_COLORS, SUPERCOMBINED_CMAP = get_combined_colors(REGION_COLORS, SUPERNUCLEUS_COLORS)

    TRINARY_CMAP = ListedColormap(['k', 'indigo', 'lightgrey'])


class MuTILsParams(object):

    # FIXME: Specify the training dataset location here
    root = opj(
        'home', 'mtageld', 'Desktop', 'cTME', 'data',
        'BootstrapNucleiManualRegions_05132021'
    )

    # dataset
    dataset_params = {
        'force_nuclear_edges': True,  # True is faithful to UNet paper
        'strong_slide_balance': False,  # False is best
        'strong_class_balance': True,  # True is best

        'original_mpp': 0.25,
        'original_side': 1024,

        'hpf_mpp': 0.5,  # 20x
        'roi_mpp': 1.0,  # 10x
        'roi_side': 256,

        'transforms': 'defaults',
        'meh_regions': ['OTHER', 'WHITE'],
    }
    train_dataset_params = deepcopy(dataset_params)
    train_dataset_params.update({
        'training': True,
        'crop_iscentral': False,
        'scale_augm_ratio': 0.1,
    })
    test_dataset_params = deepcopy(dataset_params)
    test_dataset_params.update({
        'training': False,
        'crop_iscentral': True,
        'scale_augm_ratio': None,
    })

    # dataset loader
    common_loader_kwa = {
        # 'num_workers': 0,  # sometimes (eg debug) only using 0 or 1 works
        'num_workers': 1,
        'collate_fn': collate_fn,
    }
    train_loader_kwa = deepcopy(common_loader_kwa)
    train_loader_kwa.update({
        'batch_size': 8,  # 4+ is better
        # 'shuffle': True,  # must be false if you specify sampler
    })
    test_loader_kwa = deepcopy(common_loader_kwa)
    test_loader_kwa.update({
        'batch_size': 2,  # editable config
        'shuffle': False,
    })

    # Model itself

    rcc = RegionCellCombination
    nc_regions = rcc.N_RCLASSES  # no. of region classes, no exclude
    nc_nuclei = rcc.N_NCLASSES  # no. nucl classes (incl. background), no excl.
    base_unet_params = {
        'wf': 6,  # default: 6
        'batch_norm': False,  # default: False
        'up_mode': 'upconv',  # default: upconv
    }
    roi_unet_params = deepcopy(base_unet_params)  # regions
    roi_unet_params.update({
        'in_channels': 3,  # rgb
        'depth': 5,  # editable param
    })
    hpf_unet_params = deepcopy(base_unet_params)
    hpf_unet_params.update({
        'in_channels': 3,  # rgb
        'depth': 5,  # editable param
    })
    # all params
    sf = dataset_params['roi_mpp'] / dataset_params['hpf_mpp']
    model_params = {
        'training': True,
        'hpf_mpp': dataset_params['hpf_mpp'],
        'roi_mpp': dataset_params['roi_mpp'],
        'roi_side': dataset_params['roi_side'],
        'hpf_side': dataset_params['roi_side'],  # 10x-20x
        # 'hpf_side': dataset_params['roi_side'] * sf // 2,  # 10x-40x
        'region_tumor_channel': rcc.REGION_CODES['TUMOR'] - 1,
        'region_stroma_channels': [
            rcc.REGION_CODES['STROMA'] - 1,
            rcc.REGION_CODES['TILS'] - 1],
        'nclasses_r': nc_regions,
        'nclasses_n': nc_nuclei,
        'topk_hpf': 1,
        'random_topk_hpf': True,  # if false, focus on salient stroma
        'spool_overlap': 0.25,
        'roi_unet_params': roi_unet_params,
        'roi_interm_layer': 2,  # after second upconv
        'hpf_interm_layer': 0,  # bottleneck itself
        'hpf_unet_params': hpf_unet_params,
        'transform': None,
    }

    # parametes for loss function
    loss_params = {
        'nclasses_roi': nc_regions,
        'nclasses_hpf': nc_nuclei,

        # ** Weighing BESIDE preferential sampling **

        # # Based on abundance
        # region_weights: train_dataset.ordered_region_weights,

        # Based on misclassifications we worry about
        'region_weights': [
            0.,  # EXCLUDE
            1.,  # TUMOR
            1.,  # STROMA
            1.,  # TILS
            1.,  # NORMAL
            1.,  # JUNK
            1.,  # BLOOD
            1.,  # OTHER
            1.,  # WHITE
        ],
        'nucleus_weights': [0.] + [1.] * nc_nuclei,

        # relative weighing of different loss types
        'loss_weights': {
            'roi_regions': 1.,
            'hpf_regions': 1.,
            'hpf_nuclei_pre': 1.,
            'hpf_nuclei': 1.,
        },
    }

    # optimizer
    optimizer_kws = {
        'optimizer_type': 'Adam',
        'optimizer_params': {
            'lr': 1e-4,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.0001,
        },
    }

    # learning rate scheduler
    lr_scheduler_kws = {
        'milestones': [50, 100, 200, 300],
        'gamma': 0.5,
    }
