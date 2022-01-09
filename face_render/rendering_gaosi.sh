set -x
## Example
# python -u rendering_gaosi.py --train_params_path ../video_preprocess/train1_posenew.npz \
#                              --net_params_path ../examples/test-result/test1.npz

python -u rendering_gaosi.py --train_params_path ../video_preprocessed/id00001/gangqiang_1/train_pose_new.npz \
                             --net_params_path ../examples/test-result/gangqiang_1.npz