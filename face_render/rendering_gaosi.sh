set -x
## Example
# python -u rendering_gaosi.py --train_params_path ../video_preprocess/train1_posenew.npz \
#                              --net_params_path ../examples/test-result/test1.npz

python -u rendering_gaosi.py --train_params_path ../gangqiang_video_preprocess/gangqiang_posenew.npz \
                             --net_params_path ../examples/test-result/gangqiang-3_30fps.npz