set -x

python -u fintuning2-trainheadpose_with_tb.py --audiopath ../examples/audio_preprocessed/gangqiang.pkl \
                                              --npzpath ../gangqiang_video_preprocess/gangqiang_posenew.npz \
                                              --cvspath ../gangqiang_video_preprocess/gangqiang_openface/gangqiang_512_audio.csv \
                                              --pretainpath_gen ./checkpoint/obama/Gen-20-0.0006273046686902202.mdl \
                                              --savepath ./checkpoint/debug2