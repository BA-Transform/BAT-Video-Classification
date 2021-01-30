python test_models.py \
    data/kinetics/list/kinetics_val.lst \
    checkpoints/i3d_bat.pth.tar \
    --read_mode video \
    --arch i3d_resnet50 --nonlocal_mod 2 --nltype bat --k 8 \
    --test_segments 10 --test_crops 3 --seq_length 8 --sample_rate 8 \
    -j 16