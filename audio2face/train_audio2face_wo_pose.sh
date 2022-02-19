set -x

python -u train_audio2face_wo_pose.py --audiopath ../video_preprocessed/id00001/gangqiang_2/gangqiang_2.pkl \
                                      --npzpath ../video_preprocessed/id00001/gangqiang_2/deep3dface.npz \
                                      --savepath ./checkpoint/gangqiang_2