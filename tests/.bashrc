slide_path=$1
models_path=$2

source venv/bin/activate

pip install --upgrade pip
cd /home/MuTILs_Panoptic/utils/CythonUtils
python setup.py build_ext --inplace
cd /home/MuTILs_Panoptic/histomicstk
python setup.py build_ext --inplace
cd /home

memcached -u nobody -m 64 -p 11211 -l 0.0.0.0 > /dev/null 2>&1 &
export LARGE_IMAGE_CACHE_MEMCACHED_SERVERS=127.0.0.1:11211

python MuTILs_Panoptic/tests/test_inference.py -s $slide_path -m $models_path

rm -rf /home/input/*
rm -rf /home/output/*

exit