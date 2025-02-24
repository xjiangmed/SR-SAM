rank=64
for seed in 1234 0 42 
do
CUDA_VISIBLE_DEVICES=$gpu python train.py --root_path "Data/Polyp" \
--output "output_polyp/" \
--Source_Dataset CVC-ClinicDB --Target_Dataset CVC-ColonDB ETIS Kvasir   --ckpt "Pretrained_model/sam_vit_b_01ec64.pth" \
--seed $seed --rank $rank --ema_mode --kd_weight 1e-7 \
--truncation --Dash_warm 300 --Dash_index 8 --truncation_periodic \
--suffix "_r"$rank"_seed"$seed
done