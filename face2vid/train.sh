set -x

python train.py --blink_path '../video_preprocess/train1_openface/train1_512_audio.csv' \
                --name train3 --model pose2vid --dataroot ./datasets/train3/ --netG local \
                --ngf 32 --num_D 3 --tf_log --niter_fix_global 0 --label_nc 0 --no_instance \
                --save_epoch_freq 2 --lr=0.0001 --resize_or_crop resize --no_flip --verbose --n_local_enhancers 1 --batchSize 1