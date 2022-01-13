set -x
## Example
# python test.py --audiopath ../examples/audio_preprocessed/test1.pkl \
#                --checkpath ./checkpoint/train1/Gen-90.mdl

python test.py --audiopath ../video_preprocessed/id00001/gangqiang_3/gangqiang_3.pkl \
               --checkpath ./checkpoint/gangqiang_expression_only_more_videos/Gen-50-27437.mdl \
               --use_first_gt_params False \
               --reference_gt_params ../video_preprocessed/id00002/obama_weekly_1/train_pose_new.npz