import os
from os.path import join as opj
import warnings
from typing import List, Dict
import numpy as np
import large_image
import pyvips
from glob import glob
from pandas import read_csv

from MuTILs_Panoptic.utils.GeneralUtils import load_json
from MuTILs_Panoptic.utils.MiscRegionUtils import numpy2vips
from MuTILs_Panoptic.configs import panoptic_model_configs as cfg
from MuTILs_Panoptic.utils.RegionPlottingUtils import \
    get_rgb_visualization_from_mask


class MutilsMaskVisualizer(object):
    """
    Visualize the WSI mask output from MuTILsWSIRunner by producing a WSI
    RGB mask where color indicates semantic segmentation class. Also possibly
    post the nucleus annotations to DSA.
    """
    def __init__(
        self,
        wsi_mask_and_annot_basepath: str,  # dir of wsi folders (or masks)
        *,
        slide_names: List[str] = None,  # names of slides, no file extension
        create_visualized_mask: bool = True,
        visualized_mask_dir: str = None,
        use_superclasses: bool = False,
        post_hui_annotations: bool = False,  # post annots on histomicsUI?
        slide_girder_ids: Dict[str, str] = None,
        _debug=False,
    ):
        if _debug:
            warnings.warn("Running in DEBUG mode!!!")
            raise NotImplementedError("Didn't implement debug mode yet.")

        if use_superclasses:
            raise NotImplementedError(
		"use_superclasses is not implemented, or rather, has a "
		"bug somewhere that I don't have time to fix. For now, "
		"feel free to change the config color of regions as a "
		"temporary workaround."
            )

        if post_hui_annotations:
            assert slide_girder_ids is not None, (
                "you must provide the girder ids of the WSIs on which "
                "the HistomicsUI annotations are to be posted."
            )

        if post_hui_annotations:
            warnings.warn(
                "I have not tested the annotation posting in a long time "
                "so you may need to debug this. Or it may work as-is, who "
                "knows!"
            )

        self._debug = _debug
        self.wsi_mask_and_annot_basepath = wsi_mask_and_annot_basepath
        self.slide_names = slide_names
        self.create_visualized_mask = create_visualized_mask
        self.visualized_mask_dir = visualized_mask_dir
        self.use_superclasses = use_superclasses
        self.post_hui_annotations = post_hui_annotations
        self.slide_girder_ids = slide_girder_ids

        self.slide_dirs = None
        self._maybe_set_slide_dirs()
        self._maybe_prep_girder_client()

    def run(self):
        """"""
        for slide_name in self.slide_names:
            self.visualize_wsi_mask(slide_name)
            if self.post_hui_annotations:
                self.post_hui_annots_for_slide(slide_name)

    # HELPERS -----------------------------------------------------------------

    def _maybe_prep_girder_client(self):
        """"""
        if not self.post_hui_annotations:
            return

        from MuTILs_Panoptic.configs.HistomicsAPIConfigs import StyxAPI
        self.gc = StyxAPI.connect_to_styx()

    def _maybe_set_slide_dirs(self):
        """"""
        dircontent = os.listdir(self.wsi_mask_and_annot_basepath)
        dirmasks = [j for j in dircontent if j.endswith('.tif')]
        if len(dirmasks) > 0:
            # Option 1: all wsi masks are in the same folder
            self.slide_dirs = {
                dm[:dm.rfind('.')]: self.wsi_mask_and_annot_basepath
                for dm in dirmasks
            }
        else:
            # Option 2: Each slide has a folder, with a mask inside it. This
            #  is what the output from MuTILsWSIRunner is like
            self.slide_dirs = {
                dc: opj(self.wsi_mask_and_annot_basepath, dc)
                for dc in dircontent
                if os.path.isdir(opj(self.wsi_mask_and_annot_basepath, dc))
            }

        # maybe restrict to predefined slide set
        if self.slide_names is not None:
            self.slide_dirs = {
                k: v for k, v in self.slide_dirs.items()
                if k in self.slide_names
            }
        else:
            self.slide_names = list(self.slide_dirs.keys())

    def visualize_wsi_mask(self, slide_name: str):
        """"""
        print("\n*** Visualizing mask for:", slide_name)
        # mask tile source and metadata
        tsm = large_image.getTileSource(
            opj(self.slide_dirs[slide_name], slide_name + '.tif')
        )
        mask_meta = tsm.getMetadata()

        # init visualization wsi mask
        vwsi = pyvips.Image.black(
            mask_meta['sizeX'], mask_meta['sizeY'], bands=3
        )

        roin = 0
        for tile_info in tsm.tileIterator(
                tile_size=dict(width=1024, height=1024),
                tile_overlap=dict(x=0, y=0),
                format=large_image.tilesource.TILE_FORMAT_PIL
        ):
            roin += 1
            tile_mask = np.array(tile_info['tile'])

            # only process informative tiles
            if np.max(tile_mask) == 0:
                continue

            if roin % 5 == 0:
                print(f"visualizing roi into wsi: {roin}")

            # get tile visualization
            tile_vis = get_rgb_visualization_from_mask(tile_mask)

            # replace rgb with rgb with overlay
            mtile = numpy2vips(tile_vis)
            vwsi = vwsi.insert(
                mtile, tile_info['x'], tile_info['y'], expand=True,
                background=0)

        print(f"saving visualized wsi")
        ppcm = 10 * mask_meta['mm_x']
        where = (
            self.visualized_mask_dir if self.visualized_mask_dir is not None
            else self.slide_dirs[slide_name]
        )
        vwsi.tiffsave(
            opj(where, slide_name + '_VisMask.tif'),
            tile=True, tile_width=128, tile_height=128, pyramid=True,
            compression='lzw', Q=100,
            xres=ppcm, yres=ppcm,  # yes, per cm!
        )

    def post_hui_annots_for_slide(self, slide_name):
        """"""
        print("\n*** Posting annotations for:", slide_name)
        slid = self.slide_girder_ids[slide_name]
        slide_dir = self.slide_dirs[slide_name]

        slide_meta = self.gc.get('/item/%s/tiles' % slid)
        rncd = cfg.RegionCellCombination.RNUCLEUS_CODES

        rois_per_anndoc = 25

        # nucleus colors
        ncolors = {
            grp: f"rgb({','.join(str(j) for j in color)})"
            for grp, color in cfg.VisConfigs.ALT_NUCLEUS_COLORS.items()
        }

        # go through rois
        roilocs_anndoc = {
            'name': 'roiLocs',
            'description': 'Regions of interest locations.',
            'attributes': {'algorithm_name': 'MuTILsWSIRunner'},
            "elements": []
        }
        roi_anndoc = {
            'name': 'NucleiLocs',
            'description': 'Nuclei locations.',
            'attributes': {'roi_names': []},
            'elements': []
        }
        for roin, path in enumerate(glob(opj(slide_dir, 'roi_meta', '*.json'))):
            # read metadata
            roi_meta = load_json(path)
            roi_anndoc['attributes']['roi_names'].append(roi_meta['roi_name'])

            # append roi location
            xmin, ymin = roi_meta['wsi_left'], roi_meta['wsi_top']
            xmax, ymax = roi_meta['wsi_right'], roi_meta['wsi_bottom']
            width = xmax - xmin
            height = ymax - ymin
            roilocs_anndoc['elements'].append(
                {
                    'type': 'rectangle',
                    'group': 'roi',
                    'label': {'value': roi_meta['roi_name']},
                    'lineColor': 'rgb(0,0,0)',
                    'lineWidth': 2,
                    'center': [xmin + width // 2, ymin + height // 2, 0],
                    'width': width,
                    'height': height,
                    'rotation': 0,
                }
            )

            # go through hpfs and add nuclei
            for hpf_meta in roi_meta['hpf_metas']:
                nprops = read_csv(
                    opj(slide_dir, 'hpf_nucleiProps', hpf_meta['hpf_name'] + '.csv'))
                sf = hpf_meta['mpp'] / (1e3 * slide_meta['mm_x'])
                nprops.loc[:, 'x'] = hpf_meta['wsi_left'] + sf * nprops.loc[:, 'relative_centroid_x']
                nprops.loc[:, 'y'] = hpf_meta['wsi_top'] + sf * nprops.loc[:, 'relative_centroid_y']
                for _, row in nprops.iterrows():
                    group = rncd[row['classification']]
                    roi_anndoc['elements'].append({
                        'type': 'point',
                        'group': group,
                        'label': {'value': group},
                        'lineColor': ncolors[group],
                        'lineWidth': 2,
                        'center': [int(row['x']), int(row['y']), 0],
                    })

            # maybe post and reset annotation doc
            if roin % rois_per_anndoc == 0:
                print(f"posting nuclei locs document number: %d" % roin)
                _ = self.gc.post("/annotation?itemId=" + slid, json=roi_anndoc)
                roi_anndoc['attributes']['roi_names'] = []
                roi_anndoc['elements'] = []

        # post roi location
        print(f"posting roi locations.")
        _ = self.gc.post("/annotation?itemId=" + slid, json=roilocs_anndoc)


# =============================================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize MuTILsWSIRunner output.'
    )
    parser.add_argument('-p', '--basepath', type=str)
    ARGS = parser.parse_args()

    vizer = MutilsMaskVisualizer(wsi_mask_and_annot_basepath=ARGS.basepath)
    vizer.run()
