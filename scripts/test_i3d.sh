python test_models.py \
    data/kinetics/list/kinetics_val.lst \
    checkpoints/i3d.pth.tar \
    --read_mode video \
    --arch i3d_resnet50 \
    --test_segments 10 --test_crops 3 --seq_length 8 --sample_rate 8 \
    -j 16