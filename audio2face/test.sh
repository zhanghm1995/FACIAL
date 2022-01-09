set -x
## Example
# python test.py --audiopath ../examples/audio_preprocessed/test1.pkl \
#                --checkpath ./checkpoint/train1/Gen-90.mdl

python test.py --audiopath ../video_preprocessed/id00001/gangqiang_1/gangqiang_1.pkl \
               --checkpath ./checkpoint/gangqiang_1/Gen-65.mdl \
               --use_first_gt_params True \
               --reference_gt_params ../video_preprocessed/id00001/gangqiang_1/train_pose_new.npz