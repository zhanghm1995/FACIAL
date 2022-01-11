set -x
## Example
# python test.py --audiopath ../examples/audio_preprocessed/test1.pkl \
#                --checkpath ./checkpoint/train1/Gen-90.mdl

python test.py --audiopath ../video_preprocessed/id00002/obama_weekly_2/obama_weekly_2.pkl \
               --checkpath ./checkpoint/obama_wo_pose_baseline/Gen-5-2261.mdl \
               --use_first_gt_params True \
               --reference_gt_params ../video_preprocessed/id00002/obama_weekly_1/train_pose_new.npz