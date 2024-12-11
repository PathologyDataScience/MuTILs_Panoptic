import os
from os.path import join as opj
from pandas import read_csv
import yaml

from MuTILs_Panoptic.utils.MiscRegionUtils import get_configured_logger, load_region_configs
from MuTILs_Panoptic.utils.GeneralUtils import splitlist

configuration_file = os.path.abspath(__file__).split('.')[0] + '.yaml'

class Logger:

    @staticmethod
    def set_up_logger(log_dir: str) -> object:
        """ Set up the logger for the project.

        Args:
            log_dir (str): Path to the directory where the logs will be saved.

        Returns:
            object: The logger object."""
        if not os.path.exists(log_dir):
            raise FileNotFoundError('Folder does not exist')

        LOGDIR = opj(log_dir, 'LOGS')
        os.makedirs(LOGDIR, exist_ok=True)

        logger = get_configured_logger(logdir=LOGDIR, prefix='MuTILsWSIRunner', tofile=True)
        return logger

class ConfigParser:

    @staticmethod
    def parse_config(config_file: str) -> dict:
        """Load configuration from a YAML file.

        Args:
            config_file (str): Path to the configuration file.

        Returns:
            dict: The parsed configuration as a dictionary.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """
        try:
            with open(config_file, 'r') as ymlfile:
                return yaml.load(ymlfile, Loader=yaml.FullLoader)
        except FileNotFoundError as fnfe:
            raise fnfe
        except yaml.YAMLError as ye:
            raise ye

    @staticmethod
    def check_config(config_dictionary: dict) -> None:
        """Check if the configuration values are set in the config file properly.

        Args:
            config_dictionary (dict): Configuration dictionary

        Raises:
            KeyError: If any of the required configuration values are not set in the config file.
        """
        required_keys = ['_debug', 'COHORT', 'N_SUBSETS', 'slides_path', 'base_savedir', 'model_paths',
                         'model_configs', 'restrict_to_vta', 'save_wsi_mask', 'save_annotations',
                         'save_nuclei_meta', 'save_nuclei_props', 'roi_side_hres', 'discard_edge_hres',
                         'grandch', 'gcpaths', 'topk_rois', 'topk_rois_sampling_mode',
                         'topk_salient_rois', 'vlres_scorer_kws'
        ]
        for key in required_keys:
            if key not in config_dictionary:
                raise KeyError(f'{key} not set in config file')

class RunConfigs:

    @classmethod
    def initialize(cls):
        """Initialize the configuration parameters for the pipeline.
        """
        # Load configuration parameters from the YAML file
        cls.RUN_KWARGS = ConfigParser.parse_config(configuration_file)
        ConfigParser.check_config(cls.RUN_KWARGS)

        # Set further configuration parameters
        cls.RUN_KWARGS['logger'] = Logger.set_up_logger(cls.RUN_KWARGS['base_savedir'])
        cls.RUN_KWARGS['model_configs'] = load_region_configs(cls.RUN_KWARGS['model_configs'], warn=False)

        if cls.RUN_KWARGS['grandch']:
            cls.RUN_KWARGS['logger'].info(f'Running Grand Challenge pipeline')
        else:
            cls.RUN_KWARGS['logger'].info(f'Running full cTME pipeline')

        cls.SLIDENAMES = cls.get_slide_names(cls.RUN_KWARGS)

    @staticmethod
    def get_slidenames_for_tcga(RUN_KWARGS: dict, ALL_SLIDENAMES: list) -> list:
        """Get slide names for TCGA cohort.
        NOTE: Legacy slidename splitting for TCGA. This should be replaced with a more genaral approach.

        Args:
            run_kwargs (dict): Configuration dictionary.
            all_slidenames (list): List of all slide names.

        Returns:
            list: Filtered list of slide names.
        """
        try:
            if not os.path.exists(RUN_KWARGS['data_file_path']):
                raise FileNotFoundError('PRIVATE_RSalgado_TCGA_TILScores.csv file not found')
        except KeyError:
            raise KeyError('data_file_path not set in the MuTILsWSIRunConfigs.yaml file')

        SLIDENAMES = read_csv(RUN_KWARGS['data_file_path']).iloc[:, 0].to_list()
        SLIDENAMES = [j[:12] for j in SLIDENAMES]
        return [j for j in ALL_SLIDENAMES if j[:12] in SLIDENAMES]

    @staticmethod
    def get_slidenames_for_CPS2(RUN_KWARGS: dict, ALL_SLIDENAMES: list) -> list:
        """Get slide names for CPS2 cohort.
        NOTE: Legacy slidename splitting for CPS2. This should be replaced with a more genaral approach.

        Args:
            run_kwargs (dict): Configuration dictionary.
            all_slidenames (list): List of all slide names.

        Returns:
            list: Filtered list of slide names.
        """
        try:
            if not os.path.exists(RUN_KWARGS['data_file_path']):
                raise FileNotFoundError('PSII_BRCA_FacilityIDs_20210331.csv file not found')
        except KeyError:
            raise KeyError('data_file_path not set in the MuTILsWSIRunConfigs.yaml file')

        acs_vta = read_csv(RUN_KWARGS['data_file_path'], index_col=0)
        acs_vta.rename(columns={'TILS_STR': 'vta'}, inplace=True)
        acs_vta = acs_vta.loc[:, 'vta'].map(lambda x: float(x) / 100).dropna()
        SLIDENAMES = list(acs_vta.index)
        return [j for j in ALL_SLIDENAMES if j.split('_')[0] in SLIDENAMES]

    @staticmethod
    def get_slide_names_for_NHS_breast(ALL_SLIDENAMES: list, n_subsets: int) -> list:
        """Get slide names for NHS breast cohort.

        Args:
            run_kwargs (dict): Configuration dictionary.
            all_slidenames (list): List of all slide names.

        Returns:
            list: Filtered list of slide names.
        """
        SLIDENAMES = []
        for slide_name in ALL_SLIDENAMES:
            if slide_name.endswith('.mrxs'):
                SLIDENAMES.append(slide_name)
        return splitlist(SLIDENAMES, len(SLIDENAMES) // n_subsets)


    @staticmethod
    def get_slide_names(run_kwargs: dict) -> list:
        """Get slide names based on the cohort and restrictions to
        subdivide them to run on multiple docker instances.

        Args:
            run_kwargs (dict): Configuration dictionary.

        Returns:
            list: List of slide names.
        """
        all_slidenames = os.listdir(run_kwargs['slides_path'])

        if run_kwargs['COHORT'].startswith('TCGA') and run_kwargs['restrict_to_vta']:
            return RunConfigs.get_slidenames_for_tcga(run_kwargs, all_slidenames)
        elif run_kwargs['COHORT'].endswith('CPS2') and run_kwargs['restrict_to_vta']:
            return RunConfigs.get_slidenames_for_cps2(run_kwargs, all_slidenames)
        elif run_kwargs['COHORT'] == 'NHS_breast':
            return RunConfigs.get_slide_names_for_NHS_breast(all_slidenames, run_kwargs['N_SUBSETS'])
        else:
            all_slidenames.sort()
            return splitlist(all_slidenames, len(all_slidenames) // run_kwargs['N_SUBSETS'])