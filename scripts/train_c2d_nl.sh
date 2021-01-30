python main.py \
    data/kinetics/list/kinetics_train.lst \
    data/kinetics/list/kinetics_val.lst \
    --read_mode video --resume checkpoints/resnet50.pth --soft_resume \
    --arch c2d_resnet50 --nonlocal_mod 2 \
    --num_segments 1 --seq_length 8 --sample_rate 8 \
    --lr 0.01 --lr_steps 40 80 --epochs 100 \
    -b 64 -j 48 --dropout 0.5
