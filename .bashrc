export PYTHONPATH="/home/mtageld/Desktop/MuTILs_Panoptic:$PYTHONPATH"
export PYTHONPATH="/home/mtageld/Desktop/MuTILs_Panoptic/histolab/src:$PYTHONPATH"
# ==================================================================================================
# Define the path to the first target file relative to the root
TARGET_FILE_1="/home/mtageld/Desktop/MuTILs_Panoptic/histolab/src/histolab/__init__.py"
# --------------------------------------------------------------------------------------------------
# Define the content to add to the first target file
CONTENT_1="
from .slide import *
from .masks import *
from .tiler import *
"
# --------------------------------------------------------------------------------------------------
# Check if the content already exists in the first target file
if ! grep -Fxq "from .slide import *" "$TARGET_FILE_1" || \
   ! grep -Fxq "from .masks import *" "$TARGET_FILE_1" || \
   ! grep -Fxq "from .tiler import *" "$TARGET_FILE_1"; then
    # If any line is missing, append the content to the file
    echo "$CONTENT_1" >> "$TARGET_FILE_1"
    echo "Content added to $TARGET_FILE_1."
else
    echo "Content already exists in $TARGET_FILE_1. Skipping."
fi
# ==================================================================================================
# Define the path to the second target file
TARGET_FILE_2="/home/mtageld/Desktop/MuTILs_Panoptic/histolab/src/histolab/masks.py"
# --------------------------------------------------------------------------------------------------
# Define the content to add to the second target file
CONTENT_2="from PIL import Image"
# --------------------------------------------------------------------------------------------------
CONTENT_3="
    @staticmethod
    def _remove_zero_pixels(slide):
        \"\"\"Remove zero pixels from the thumbnail.\"\"\"
        thumb = np.array(slide.thumbnail)
        mask = np.all(thumb == 0, axis=-1)
        thumb[mask] = [255, 255, 255]
        return Image.fromarray(thumb)
"
# --------------------------------------------------------------------------------------------------
CONTENT_4="    def _thumb_mask(self, slide, thumb=None):"
# --------------------------------------------------------------------------------------------------
CONTENT_5="        thumb = self._remove_zero_pixels(slide)"
# --------------------------------------------------------------------------------------------------
# Check if the new content already exists in the second target file
if ! grep -Fxq "from PIL import Image" "$TARGET_FILE_2"; then
    sed -i "22r /dev/stdin" "$TARGET_FILE_2" <<< "$CONTENT_2"
    sed -i "72r /dev/stdin" "$TARGET_FILE_2" <<< "$CONTENT_3"
    sed -i "83,84d" "$TARGET_FILE_2"
    sed -i "82r /dev/stdin" "$TARGET_FILE_2" <<< "$CONTENT_4"
    sed -i "85r /dev/stdin" "$TARGET_FILE_2" <<< "$CONTENT_5"
else
    echo "Content already exists in $TARGET_FILE_2. Skipping."
fi