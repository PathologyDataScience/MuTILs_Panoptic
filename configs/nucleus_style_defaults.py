from pandas import read_csv
from collections import OrderedDict
import os
from os.path import join as opj

from MuTILs_Panoptic.utils.GeneralUtils import ordered_vals_from_ordered_dict


CONFIGPATH = os.path.dirname(__file__)


class DefaultAnnotationStyles(object):

    DEFAULT_LINEWIDTH = 3.0
    DEFAULT_OPACITY = 0

    # FOV styles
    FOV_STYLES = {
        "fov_preapproved": {"lineColor": "rgb(255,10,255)"},
        "fov_for_programmatic_edit": {"lineColor": "rgb(255,0,0)"},
        "fov_basic": {"lineColor": "rgb(0,0,0)"},
        "fov_representative": {"lineColor": "rgb(255,255,255)"},
        "fov_problematic": {"lineColor": "rgb(255,255,0)"},
        "fov_discordant": {"lineColor": "rgb(0,0,0)"},
    }
    for k in FOV_STYLES.keys():
        FOV_STYLES[k]['type'] = 'rectangle'

    # *** Standard styles for final dataset ***
    STANDARD_STYLES = {
        "tumor": {"lineColor": "rgb(255,0,0)"},
        "mitotic_figure": {"lineColor": "rgb(255,191,0)"},
        "fibroblast": {"lineColor": "rgb(0,230,77)"},
        "lymphocyte": {"lineColor": "rgb(0,0,255)"},
        "plasma_cell": {"lineColor": "rgb(0,255,255)"},
        "macrophage": {"lineColor": "rgb(51,102,153)"},
        "neutrophil": {"lineColor": "rgb(51,102,204)"},
        "eosinophil": {"lineColor": "rgb(128,0,0)"},
        "apoptotic_body": {"lineColor": "rgb(255,128,0)"},
        "vascular_endothelium": {"lineColor": "rgb(102,0,204)"},
        "myoepithelium": {"lineColor": "rgb(250,50,250)"},
        "ductal_epithelium": {"lineColor": "rgb(255,128,255)"},
        "unlabeled": {"lineColor": "rgb(80,80,80)"},
    }
    for k in STANDARD_STYLES.keys():
        STANDARD_STYLES[k]['type'] = "polyline"

    # standard class names and colors in plt-compatible format
    MAIN_CLASSES = ['tumor', 'fibroblast', 'lymphocyte']
    CLASSES = list(STANDARD_STYLES.keys())
    COLORS = {
        c: [
            int(v) / 255 for v in
            s['lineColor'].split('rgb(')[1].split(')')[0].split(',')
        ] for c, s in STANDARD_STYLES.items()
    }
    COLORS['all'] = COLORS['unlabeled']
    COLORS['detection'] = COLORS['unlabeled']
    COLORS['classification'] = COLORS['unlabeled']

    STANDARD_STYLES.update(FOV_STYLES)  # standard styles incl. fovs

    # assign other default annotation document attributes
    for stylename in STANDARD_STYLES.keys():

        style = STANDARD_STYLES[stylename]

        # by default, group and label are assigned dict key
        STANDARD_STYLES[stylename]["group"] = stylename
        STANDARD_STYLES[stylename]["label"] = {"value": stylename}

        # common defaults
        STANDARD_STYLES[stylename]["lineWidth"] = DEFAULT_LINEWIDTH
        fillColor = style['lineColor']
        fillColor = fillColor.replace("rgb", "rgba")
        fillColor = fillColor[:fillColor.rfind(")")] + ",{})".format(
            DEFAULT_OPACITY)

        # type-specific defaults
        if STANDARD_STYLES[stylename]["type"] == "rectangle":
            STANDARD_STYLES[stylename]['center'] = [0, 0, 0]
            STANDARD_STYLES[stylename]['height'] = 1
            STANDARD_STYLES[stylename]['width'] = 1
            STANDARD_STYLES[stylename]['rotation'] = 0
            STANDARD_STYLES[stylename]['normal'] = [0, 0, 1]
            STANDARD_STYLES[stylename]["lineWidth"] = 6

        elif STANDARD_STYLES[stylename]["type"] == "polyline":
            STANDARD_STYLES[stylename]['closed'] = True
            STANDARD_STYLES[stylename]['points'] = [
                [0, 0, 0], [0, 1, 0], [1, 0, 0]]
            STANDARD_STYLES[stylename]["fillColor"] = fillColor

    # GT codes dict for parsing into label mask
    GTCODE_PATH = opj(CONFIGPATH, 'nucleus_GTcodes.csv')
    gtcodes_df = read_csv(GTCODE_PATH)
    gtcodes_df.index = gtcodes_df.loc[:, 'group']
    gtcodes_dict = gtcodes_df.to_dict(orient='index')

    # reverse dict for quick access (indexed by GTcode in mask)
    # NOTE: since multiple fov styles share the same GTcode, they map to
    # the same key as 'fov_discordant'
    rgtcodes_dict = {v['GT_code']: v for k, v in gtcodes_dict.items()}


class NucleusCategories(object):

    das = DefaultAnnotationStyles

    # ground truth codes as they appear in masks
    gtcodes_dict = das.gtcodes_dict
    rgtcodes_dict = das.rgtcodes_dict

    # categories that should only be used to TRAIN detector and which have
    # no classification because of their ambiguity and high discordance
    ambiguous_categs = [
        'apoptotic_body',
        'unlabeled',
    ]
    ambiguous_categs.extend(
        [f'correction_{v}' for v in ambiguous_categs]
    )

    # map from raw categories to main categories to be learned
    raw_to_main_categmap = OrderedDict({
        'tumor': 'tumor_nonMitotic',
        'mitotic_figure': 'tumor_mitotic',
        'fibroblast': 'nonTILnonMQ_stromal',
        'vascular_endothelium': 'nonTILnonMQ_stromal',
        'macrophage': 'macrophage',
        'lymphocyte': 'lymphocyte',
        'plasma_cell': 'plasma_cell',
        'neutrophil': 'other_nucleus',
        'eosinophil': 'other_nucleus',
        'myoepithelium': 'other_nucleus',
        'ductal_epithelium': 'other_nucleus',
    })
    raw_to_main_categmap.update({
        f'correction_{k}': v for k, v in raw_to_main_categmap.items()
    })
    raw_to_main_categmap.update({k: 'AMBIGUOUS' for k in ambiguous_categs})
    raw_categs = raw_to_main_categmap.keys()

    # map from main categories to super-categories
    main_to_super_categmap = OrderedDict({
        'tumor_nonMitotic': 'tumor_any',
        'tumor_mitotic': 'tumor_any',
        'nonTILnonMQ_stromal': 'nonTIL_stromal',
        'macrophage': 'nonTIL_stromal',
        'lymphocyte': 'sTIL',
        'plasma_cell': 'sTIL',
        'other_nucleus': 'other_nucleus',
        'AMBIGUOUS': 'AMBIGUOUS',
    })

    # same but from main to supercategs
    raw_to_super_categmap = OrderedDict()
    for k, v in raw_to_main_categmap.items():
        raw_to_super_categmap[k] = main_to_super_categmap[v]

    # names & *contiguous* gt codes for main categories
    main_categs = ordered_vals_from_ordered_dict(raw_to_main_categmap)
    main_categs_codes = {j: i + 1 for i, j in enumerate(main_categs)}

    # names & *contiguous* gt codes for super categories
    super_categs = ordered_vals_from_ordered_dict(main_to_super_categmap)
    super_categs_codes = {j: i + 1 for i, j in enumerate(super_categs)}

    # direct dict mapping from main categs to super category gt codes
    main_codes_to_super_codes = {}
    for k, v in main_categs_codes.items():
        main_codes_to_super_codes[v] = super_categs_codes[
            main_to_super_categmap[k]]

    # direct dict mapping from raw categs to main category gt codes
    raw_to_main_categs_codes = {}
    for k, v in raw_to_main_categmap.items():
        raw_to_main_categs_codes[k] = main_categs_codes[v]

    # direct dict mapping from raw categs to super category gt codes
    raw_to_super_categs_codes = {}
    for k, v in raw_to_super_categmap.items():
        raw_to_super_categs_codes[k] = super_categs_codes[v]

    # map from raw categories to PURE detection categories
    raw_to_puredet_categmap = OrderedDict()
    for k in raw_to_main_categmap.keys():
        raw_to_puredet_categmap[k] = \
            'nucleus' if k not in ambiguous_categs else 'AMBIGUOUS'

    # names & *contiguous* gt codes for PURE detection
    puredet_categs = ordered_vals_from_ordered_dict(raw_to_puredet_categmap)
    puredet_categs_codes = {j: i + 1 for i, j in enumerate(puredet_categs)}
    raw_to_puredet_categs_codes = {}
    for k, v in raw_to_puredet_categmap.items():
        raw_to_puredet_categs_codes[k] = puredet_categs_codes[v]

    # categmap from ORIGINAL codes (what the participants SAW)
    # we do this at the level of super-classes because there's no fine
    # distinction at the crude level of data shown
    original_to_super_categmap = OrderedDict({
        # tumor
        'tumor': 'tumor_any',
        # stromal
        'fibroblast': 'nonTIL_stromal',
        'vascular_endothelium': 'nonTIL_stromal',
        # tils
        'lymphocyte': 'sTIL',
        'plasma_cell': 'sTIL',
        'other_inflammatory': 'sTIL',
        # other
        'normal_acinus_or_duct': 'other_nucleus',
        'nerve': 'other_nucleus',
        'skin_adnexa': 'other_nucleus',
        'other': 'other_nucleus',
        'adipocyte': 'other_nucleus',
        # meh / junk
        'necrosis_or_debris': 'AMBIGUOUS',
        'glandular_secretions': 'AMBIGUOUS',
        'blood': 'AMBIGUOUS',
        'exclude': 'AMBIGUOUS',
        'metaplasia_NOS': 'AMBIGUOUS',
        'mucoid_material': 'AMBIGUOUS',
        'lymphatics': 'AMBIGUOUS',
        'undetermined': 'AMBIGUOUS',
    })

    # categmap from REGION codes (what participants 'consulted' while
    # making their assessment) and super-class code.
    regions_to_super_categmap = OrderedDict({
        # tumor
        'tumor': 'tumor_any',
        'angioinvasion': 'tumor_any',
        'dcis': 'tumor_any',
        # stromal
        'stroma': 'nonTIL_stromal',
        'blood_vessel': 'nonTIL_stromal',
        # tils
        'lymphocytic_infiltrate': 'sTIL',
        'plasma_cells': 'sTIL',
        'other_immune_infiltrate': 'sTIL',
        # other
        'normal_acinus_or_duct': 'other_nucleus',
        'nerve': 'other_nucleus',
        'skin_adnexa': 'other_nucleus',
        'other': 'other_nucleus',
        'fat': 'other_nucleus',
        # meh / junk
        'necrosis_or_debris': 'AMBIGUOUS',
        'glandular_secretions': 'AMBIGUOUS',
        'blood': 'AMBIGUOUS',
        'exclude': 'AMBIGUOUS',
        'metaplasia_NOS': 'AMBIGUOUS',
        'mucoid_material':  'AMBIGUOUS',
        'lymphatics': 'AMBIGUOUS',
        'undetermined': 'AMBIGUOUS',
    })
    regions_to_super_categs_codes = {}
    for k, v in regions_to_super_categmap.items():
        regions_to_super_categs_codes[k] = super_categs_codes[v]
