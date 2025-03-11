slide_path=$1
models_path=$2

source venv/bin/activate

python MuTILs_Panoptic/tests/test_inference.py -s $slide_path -m $models_path

rm -rf /home/input/*
rm -rf /home/output/*

exit