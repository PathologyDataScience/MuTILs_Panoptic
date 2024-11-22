import os
from os.path import join as opj
from pandas import read_csv
import yaml

from MuTILs_Panoptic.utils.MiscRegionUtils import get_configured_logger, load_region_configs
from MuTILs_Panoptic.utils.GeneralUtils import splitlist

class Logger:
    '''
    Set up the logger for the project.

    Parameters:
        target_folder (str): The target folder to save the logs.

    Returns:
        object: The logger object.

    Raises:
        FileNotFoundError: If the target folder does not exist.
    '''
    def __init__(self, target_folder: str) -> None:
        self.target_folder = target_folder

    def set_up_logger(self) -> object:
        '''
        Set up the logger for the project.

        Returns:
            object: The logger object.
        '''
        if not os.path.exists(self.target_folder):
            raise FileNotFoundError("Target folder does not exist")

        LOGDIR = opj(self.target_folder, 'LOGS')
        os.makedirs(LOGDIR, exist_ok=True)

        logger = get_configured_logger(
            logdir=LOGDIR, prefix='MuTILsWSIRunner', tofile=True
        )

        return logger

class ParseConfigs:
    '''
    Parse configuration parameters from the configuration file.

    Parameters:
        config_file (str): The path to the configuration file.

    Attributes:
        config_dict (dict): The configuration dictionary.
        logger (object): The logger object to collect logs and errors.
        DEBUG (bool): The debug mode flag.
        MODELNAME (str): The model name.
        COHORT (str): The cohort name.
        N_SUBSETS (int): The number of subsets to divide the cohort into.
        BASEPATH (str): The base path of the project directory.
        INPUT_PATH (str): The input path.
        OUTPUT_PATH (str): The output path.
        BASEMODELPATH (str): The base model path.
        restrict_to_vta (bool): The restrict to VTA flag.
        save_wsi_mask (bool): The save WSI mask flag.
        save_annotations (bool): The save annotations flag.
        save_nuclei_meta (bool): The save nuclei meta flag.
        save_nuclei_props (bool): The save nuclei props flag.
        grandch (bool): The grand challenge flag.
        gcpaths (dict): The grand challenge paths.
        topk_rois (int): The top k ROIs.
        topk_rois_sampling_mode (str): The top k ROIs sampling mode.
        topk_salient_rois (int): The top k salient ROIs.
        vlres_scorer_kws (dict): The VLRes scorer keywords.

    Raises:
        KeyError: If any of the required configuration values are not set in the config file.
    '''
    def __init__(self, logger: object, config_file: str) -> None:
        self.logger = logger
        self.config_dict = self.load_config(config_file)
        self.set_config_values()

    def load_config(self, file_path: str) -> dict:
        """
        Load configuration from a YAML file.

        Parameters:
            file_path (str): The path to the YAML configuration file.

        Returns:
            dict: The loaded configuration as a dictionary.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """
        try:
            with open(file_path, 'r') as ymlfile:
                return yaml.load(ymlfile, Loader=yaml.FullLoader)
        except FileNotFoundError as e:
            self.logger.error(f"Config file not found: {file_path}")
            raise e
        except yaml.YAMLError as exc:
            self.logger.error(f"Error parsing config file: {file_path}")
            raise exc

    def set_config_values(self) -> None:
        '''
        Set the configuration values from the config file to the class attributes.

        Raises:
            KeyError: If any of the required configuration values are not set in the config file.
        '''
        required_keys = ['DEBUG', 'MODELNAME', 'COHORT', 'N_SUBSETS',
                         'BASEPATH', 'INPUT_PATH', 'OUTPUT_PATH', 'BASEMODELPATH',
                         'restrict_to_vta', 'save_wsi_mask', 'save_annotations',
                         'save_nuclei_meta', 'save_nuclei_props',
                         'grandch', 'gcpaths', 'topk_rois', 'topk_rois_sampling_mode',
                         'topk_salient_rois', 'vlres_scorer_kws']
        for key in required_keys:
            if key not in self.config_dict:
                self.logger.error(f"{key} not set in config file")
                raise KeyError(f"{key} not set in config file")

        self.DEBUG = self.config_dict['DEBUG']
        self.MODELNAME = self.config_dict['MODELNAME']
        self.COHORT = self.config_dict['COHORT']
        self.N_SUBSETS = self.config_dict['N_SUBSETS']
        # self.N_SUBSETS = 8 if not self.DEBUG else 1
        # self.N_SUBSETS = 16 if not self.DEBUG else 1
        self.BASEPATH = self.config_dict['BASEPATH']
        self.INPUT_PATH = opj(self.config_dict['INPUT_PATH'], self.COHORT)
        self.OUTPUT_PATH = opj(self.config_dict['OUTPUT_PATH'], self.COHORT)
        self.BASEMODELPATH = self.config_dict['BASEMODELPATH']

        self.restrict_to_vta = self.config_dict['restrict_to_vta']
        self.save_wsi_mask = self.config_dict['save_wsi_mask']
        self.save_annotations = self.config_dict['save_annotations']
        self.save_nuclei_meta = self.config_dict['save_nuclei_meta']
        self.save_nuclei_props = self.config_dict['save_nuclei_props']

        self.grandch = self.config_dict['grandch']
        self.topk_rois = self.config_dict['topk_rois']
        self.topk_rois_sampling_mode = self.config_dict['topk_rois_sampling_mode']
        self.topk_salient_rois = self.config_dict['topk_salient_rois']

        if self.grandch:
            self.gcpaths = {
                'roilocs_in': opj(self.INPUT_PATH, self.config_dict['gcpaths']['roilocs_in']),
                'cta2vta': opj(self.BASEPATH, self.config_dict['gcpaths']['cta2vta']),
                'roilocs_out': opj(self.OUTPUT_PATH, self.config_dict['gcpaths']['roilocs_out']),
                'result_file': opj(self.OUTPUT_PATH, self.config_dict['gcpaths']['result_file']),
                'tilscore_file': opj(self.OUTPUT_PATH, self.config_dict['gcpaths']['tilscore_file']),
                'detect_file': opj(self.OUTPUT_PATH, self.config_dict['gcpaths']['detect_file']),
                'wsi_mask': opj(self.OUTPUT_PATH, 'images', self.config_dict['gcpaths']['wsi_mask']),
            }
            self.vlres_scorer_kws = {
                'check_tissue': self.config_dict['vlres_scorer_kws']['check_tissue'],
                'tissue_percent': self.config_dict['vlres_scorer_kws']['tissue_percent'],
                'pixel_overlap': self.config_dict['vlres_scorer_kws']['pixel_overlap'],
            }
            self.logger.info(f"Running Grand Challenge pipeline")
        else:
            self.gcpaths = self.config_dict['gcpaths']
            self.vlres_scorer_kws = self.config_dict['vlres_scorer_kws']
            self.logger.info(f"Running full cTME pipeline")

class RunConfigs:

    current_file_path = os.path.dirname(os.path.abspath(__file__))
    config_file_path = opj(current_file_path, 'config.yaml')

    logger = Logger(current_file_path).set_up_logger()

    # IMPORTANT: THIS SWITCHES THE ANALYSIS TYPE in the config.yaml file
    cfg = ParseConfigs(logger, config_file_path)

    # model weights
    MODEL_PATHS = {}
    for f in range(1, 6):
        MODEL_PATHS[f'{cfg.MODELNAME}-fold{f}'] = opj(
            cfg.BASEMODELPATH, f'fold_{f}', f'{cfg.MODELNAME}_fold{f}.pt'
        )

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Subdividing cohort to run on multiple docker instances

    ALL_SLIDENAMES = os.listdir(cfg.INPUT_PATH)

    if cfg.COHORT.startswith('TCGA') and cfg.restrict_to_vta:
        # RESTRICT TO ROBERTO SALGADO ASSESSED SLIDES
        SLIDENAMES = read_csv(opj(
            cfg.BASEPATH, 'data', 'tcga-clinical',
            'PRIVATE_RSalgado_TCGA_TILScores.csv'
        )).iloc[:, 0].to_list()
        SLIDENAMES = [j[:12] for j in SLIDENAMES]
        SLIDENAMES = [j for j in ALL_SLIDENAMES if j[:12] in SLIDENAMES]

    elif cfg.COHORT.endswith('CPS2') and cfg.restrict_to_vta:
        # RESTRICT TO TED ASSESSED SLIDES (CPS2)
        acs_vta = read_csv(
            opj(cfg.BASEPATH, 'data', 'acs-clinical',
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
            SLIDENAMES, len(SLIDENAMES) // cfg.N_SUBSETS
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
        # roi_side_hres=512 if ParseConfigs.DEBUG else 1024,
        roi_side_hres=1024,
        discard_edge_hres=0,  # keep 0 -> slow + can't get the gap to be exact
        logger=cfg.logger,

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

        _debug=cfg.DEBUG,
    )
