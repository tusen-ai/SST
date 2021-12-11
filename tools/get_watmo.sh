PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 tools/create_data.py waymo --root-path ./data/waymo/ --out-dir /data2/waymo_mmdet/ --workers 32 --extra-tag waymo
