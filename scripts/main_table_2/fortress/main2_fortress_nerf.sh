cd NeRF

python run_nerf.py \
    --config configs/llff_data/fortress.txt \
    --expname $(basename "${0%.*}") \
    --chunk 8192 \
    --N_rand 1024 \
    --run_without_colmap both \
    --N_iters 600001 \
    --lrate_decay 300
