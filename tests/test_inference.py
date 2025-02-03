import os
import subprocess
import argparse

# Color codes for printing
RED = '\033[31m'
GREEN = '\033[32m'
RESET = '\033[0m'

class FileCopier:

    @staticmethod
    def copy_test_slide(slide: str):
        """Copies a test slide to the input folder.

        Parameters:
        ------------
            slide (str): Path to the slide file.
        """
        print("Copying slide to the input folder...")
        rsync_slide = [
            "rsync",
            "--info=progress2",
            slide,
            "/home/input/."
        ]
        subprocess.run(rsync_slide, check=True)

        # If MRXS slide, copy the accompanying files
        if slide.endswith(".mrxs"):

            slide_name = os.path.splitext(os.path.basename(slide))[0]
            subprocess.run(["mkdir", "-p", f"/home/input/{slide_name}"])

            slide_folder = slide.split('.')[0]

            rsync_files = [
                "rsync",
                "--info=progress2",
                "-r",
                slide_folder+"/.",
                f"/home/input/{slide_name}/."
            ]
            subprocess.run(rsync_files, check=True)

        chmod = [
            "chmod",
            "-R",
            "777",
            "/home/input"
        ]
        subprocess.run(chmod, check=True)
        print("Slide copied successfully!")

    @staticmethod
    def copy_models(models: str):
        """Copies models to the models folder.

        Parameters:
        ------------
            models (str): Path to the models folder.
        """
        print("Copying models to the models folder...")

        subprocess.run(["mkdir", "-p", "/home/models"])

        rsync_models = [
            "rsync",
            "--info=progress2",
            "-r",
            "--partial",
            "--append-verify",
            models,
            "/home/models/."
        ]
        subprocess.run(rsync_models, check=True)
        print("Models copied successfully!")

class OutputTester:

    @staticmethod
    def _check_folder(path: str, name: str) -> bool:
        """Checks if a folder exists and prints the result.

        Parameters:
        ------------
            path (str): Path to the folder.
            name (str): Name of the folder.

        Returns:
        --------
            bool: True if the folder exists, False otherwise.
        """
        if os.path.exists(path):
            print(f"{GREEN}OK.{RESET} {name} folder exists.")
            return True
        else:
            print(f"{RED}Error:{RESET} {name} folder is missing.")
            return False

    @staticmethod
    def _check_files(folder_path: str, expected_count: int, folder_name: str) -> list:
        """Checks if a folder contains the expected number of files and prints results.

        Parameters:
        ------------
            folder_path (str): Path to the folder.
            expected_count (int): Expected number of files.
            folder_name (str): Name of the folder.

        Returns:
        --------
            list: List of files in the folder.
        """
        files = os.listdir(folder_path)
        if len(files) == expected_count:
            print(f"{GREEN}OK.{RESET} {folder_name} files exist.")
        else:
            print(f"{RED}Error:{RESET} {folder_name} files are missing.")

        return files

    @staticmethod
    def _check_file_size(folder_path: str, files: list):
        """Checks if files in a folder are empty and prints results.

        Parameters:
        ------------
            folder_path (str): Path to the folder.
            files (list): List of files in the folder.
        """
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.getsize(file_path) < 1000:
                print(f"{RED}Error:{RESET} {file} is empty.")
            else:
                print(f"{GREEN}OK.{RESET} {file} is not empty.")

    @classmethod
    def test_results(cls, slide: str):
        """Tests if the expected output files and folders exist.

        Parameters:
        ------------
            slide (str): Path to the slide file.
        """
        print("Testing the outputs...")

        output_base = "/home/output/perSlideResults"
        if not cls._check_folder(output_base, "perSlideResults"):
            return  # Stop checking if the base folder is missing

        slide_name = os.path.splitext(os.path.basename(slide))[0]
        slide_folder = os.path.join(output_base, slide_name)
        if not cls._check_folder(slide_folder, slide_name):
            return  # Stop checking if slide folder is missing

        # List of folders and expected file counts
        folders_to_check = {
            "annotations": 7,
            "nucleiMeta": 2,
            "nucleiProps": 2,
            "roiMasks": 2,
            "roiMeta": 2,
            "roiVis": 2
        }

        # Iterate over each folder and perform checks
        for folder_name, expected_count in folders_to_check.items():
            folder_path = os.path.join(slide_folder, folder_name)
            if cls._check_folder(folder_path, folder_name):
                files = cls._check_files(folder_path, expected_count, folder_name)
                cls._check_file_size(folder_path, files)

        # Check specific output files
        output_files = [
            (f"{slide_name}_RAGraph.png", "RAGraph image"),
            (f"{slide_name}_RoiLocs.csv", "RoiLocs CSV"),
            (f"{slide_name}_RoiLocs.png", "RoiLocs PNG"),
            (f"{slide_name}.json", "Aggregated results JSON"),
            (f"{slide_name}.tif", "Segmentation image"),
        ]

        for file_name, description in output_files:
            file_path = os.path.join(slide_folder, file_name)
            if os.path.exists(file_path):
                print(f"{GREEN}OK.{RESET} {description} exists.")
                if os.path.getsize(file_path) < 1000:
                    print(f"{RED}Error:{RESET} {file_name} is empty.")
                else:
                    print(f"{GREEN}OK.{RESET} {file_name} is not empty.")
            else:
                print(f"{RED}Error:{RESET} {description} is missing.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy a test slide to the input folder.")
    parser.add_argument("-s", "--slide", type=str, help="Path to the slide file")
    parser.add_argument("-m", "--models", type=str, help="Path to the models folder")
    return parser.parse_args()

def run_mutils():
    """Runs the MuTILsWSIRunner.py script."""
    print("Running MuTILsWSIRunner.py...")
    run_mutils = [
        "python",
        "MuTILs_Panoptic/mutils_panoptic/MuTILsWSIRunner.py",
    ]
    subprocess.run(run_mutils, check=True)
    print("MuTILsWSIRunner.py finished successfully!")

if __name__ == "__main__":

    args = parse_args()

    FileCopier.copy_test_slide(args.slide)
    FileCopier.copy_models(args.models)
    run_mutils()
    OutputTester.test_results(args.slide)