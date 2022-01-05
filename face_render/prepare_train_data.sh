set -x
# python -u handle_netface.py --param_folder ../gangqiang_video_preprocess/gangqiang_deep3Dface

# python -u fit_headpose.py --csv_path ../gangqiang_video_preprocess/gangqiang_openface/gangqiang_512_audio.csv \
#                           --deepface_path ../gangqiang_video_preprocess/gangqiang_deep3Dface/train1.npz \
#                           --save_path ../gangqiang_video_preprocess/gangqiang_posenew.npz

python -u render_netface_fitpose.py --real_params_path ../gangqiang_video_preprocess/gangqiang_posenew.npz \
                                    --outpath ../gangqiang_video_preprocess/train_A/