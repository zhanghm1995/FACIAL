set -x
## Example
# python test.py --audiopath ../examples/audio_preprocessed/test1.pkl \
#                --checkpath ./checkpoint/train1/Gen-90.mdl

python test.py --audiopath ../examples/audio_preprocessed/gangqiang-3_30fps.pkl \
               --checkpath ./checkpoint/debug/Gen-160.mdl \
               --use_first_gt_params True