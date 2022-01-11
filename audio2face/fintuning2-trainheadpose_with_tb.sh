set -x

# python -u fintuning2-trainheadpose_with_tb.py --audiopath ../examples/audio_preprocessed/gangqiang.pkl \
#                                               --npzpath ../gangqiang_video_preprocess/gangqiang_posenew.npz \
#                                               --cvspath ../gangqiang_video_preprocess/gangqiang_openface/gangqiang_512_audio.csv \
#                                               --pretainpath_gen ./checkpoint/obama/Gen-20-0.0006273046686902202.mdl \
#                                               --savepath ./checkpoint/debug2

# python -u fintuning2-trainheadpose_with_tb.py --audiopath ../examples/audio_preprocessed/train1.pkl \
#                                               --npzpath ../video_preprocess/train1_posenew.npz \
#                                               --cvspath ../video_preprocess/train1_openface/train1_512_audio.csv \
#                                               --pretainpath_gen ./checkpoint/obama/Gen-20-0.0006273046686902202.mdl \
#                                               --savepath ./checkpoint/train1_with_tb

## For not using OpenFace case
# python -u fintuning2-trainheadpose_with_tb.py --audiopath ../video_preprocessed/id00001/gangqiang_1/gangqiang_1.pkl \
#                                               --npzpath ../video_preprocessed/id00001/gangqiang_1/train_pose_new.npz \
#                                               --pretainpath_gen ./checkpoint/chinese/Gen-105.mdl \
#                                               --savepath ./checkpoint/gangqiang_1

# ./checkpoint/gangqiang_1_with_eval_wo_pose/Gen-15-1983.mdl

# python -u fintuning2-trainheadpose_with_tb.py --audiopath ../video_preprocessed/id00001/gangqiang_1/gangqiang_1.pkl \
#                                               --npzpath ../video_preprocessed/id00001/gangqiang_1/train_pose_new.npz \
#                                               --eval_audiopath ../video_preprocessed/id00001/gangqiang_3/gangqiang_3.pkl \
#                                               --eval_npzpath ../video_preprocessed/id00001/gangqiang_3/train_pose_new.npz \
#                                               --pretainpath_gen ./checkpoint/obama/Gen-20-0.0006273046686902202.mdl \
#                                               --savepath ./checkpoint/gangqiang_1_with_eval_wo_pose_60_frames_with_dropout


python -u fintuning2-trainheadpose_with_tb.py --audiopath ../video_preprocessed/id00002/obama_weekly_1/obama_weekly_1.pkl \
                                              --npzpath ../video_preprocessed/id00002/obama_weekly_1/train_pose_new.npz \
                                              --eval_audiopath ../video_preprocessed/id00002/obama_weekly_2/obama_weekly_2.pkl \
                                              --eval_npzpath ../video_preprocessed/id00002/obama_weekly_2/train_pose_new.npz \
                                              --pretainpath_gen ./checkpoint/obama/Gen-20-0.0006273046686902202.mdl \
                                              --savepath ./checkpoint/obama_wo_pose_baseline