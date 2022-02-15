set -x

# python test_video.py --test_id_name test1 --blink_path '../examples/test-result/test1.npz' \
#                      --name train3 --model pose2vid --dataroot ./datasets/train3/ \
#                      --which_epoch latest --netG local --ngf 32 --label_nc 0 --n_local_enhancers 1 \
#                      --no_instance --resize_or_crop resize


## Run 
# python test_video.py --test_id_name gangqiang2 --blink_path '../examples/test-result/obama2.npz' \
#                      --name gangqiang2 --model pose2vid --dataroot ./datasets/train3/ \
#                      --which_epoch latest --netG local --ngf 32 --label_nc 0 --n_local_enhancers 1 \
#                      --no_instance --resize_or_crop resize


python test_video.py --test_id_name gangqiang5 --blink_path '../examples/test-result/obama2.npz' \
                     --name gangqiang2 --model pose2vid --dataroot ./datasets/train3/ \
                     --which_epoch latest --netG local --ngf 32 --label_nc 0 --n_local_enhancers 1 \
                     --no_instance --resize_or_crop resize