# python main_pretrain.py --num_bootstrap 2 --use_new_feature_predictor
# python main_pretrain.py --num_bootstrap 4 --use_new_feature_predictor
# python main_pretrain.py --num_bootstrap 5 --use_new_feature_predictor
# python main_pretrain.py --num_bootstrap 8 --use_new_feature_predictor
# python main_pretrain.py --num_bootstrap 10 --use_new_feature_predictor
# python main_pretrain.py --num_bootstrap 20 --use_new_feature_predictor
#
# python main_pretrain.py --num_bootstrap 4 --feature_class hog --blr 1e-2
# python main_pretrain.py --num_bootstrap 5 --feature_class hog --blr 1e-2
#
# python main_pretrain.py --enable_ema --ema_decay 0.9
# python main_pretrain.py --enable_ema --ema_decay 0.99
# python main_pretrain.py --enable_ema --ema_decay 0.999
#
# python main_finetune.py --finetune checkpoints/pretrain-ema-9.pth
# python main_finetune.py --finetune checkpoints/pretrain-ema-99.pth
# python main_finetune.py --finetune checkpoints/pretrain-ema-999.pth
# python main_finetune.py --finetune checkpoints/pretrain-ema-9999.pth
#
# python main_linprobe.py --finetune checkpoints/pretrain-k1.pth
# python main_linprobe.py --finetune checkpoints/pretrain-ema-9.pth
# python main_linprobe.py --finetune checkpoints/pretrain-ema-99.pth
# python main_linprobe.py --finetune checkpoints/pretrain-ema-999.pth
# python main_linprobe.py --finetune checkpoints/pretrain-ema-9999.pth

# python main_pretrain.py --enable_ema --target_layer_index 6 --ema_decay 0.9
# python main_pretrain.py --enable_ema --target_layer_index 6 --ema_decay 0.99
# python main_pretrain.py --enable_ema --target_layer_index 6 --ema_decay 0.999
# python main_pretrain.py --enable_ema --target_layer_index 6 --ema_decay 0.9999

# python main_pretrain.py --num_bootstrap 2 --target_layer_index 6 --use_new_feature_predictor
# python main_pretrain.py --num_bootstrap 4 --target_layer_index 6 --use_new_feature_predictor
# python main_pretrain.py --num_bootstrap 5 --target_layer_index 6 --use_new_feature_predictor
# python main_pretrain.py --num_bootstrap 8 --target_layer_index 6 --use_new_feature_predictor
# python main_pretrain.py --num_bootstrap 10 --target_layer_index 6 --use_new_feature_predictor

# python main_pretrain.py --num_bootstrap 1 --target_layer_index 11

# python main_pretrain.py --num_bootstrap 2  --target_layer_index 5
# python main_pretrain.py --num_bootstrap 4  --target_layer_index 5
# python main_pretrain.py --num_bootstrap 5  --target_layer_index 5
# python main_pretrain.py --num_bootstrap 8  --target_layer_index 5

# python main_pretrain.py --num_bootstrap 2  --target_layer_index 11
# python main_pretrain.py --num_bootstrap 4  --target_layer_index 11
# python main_pretrain.py --num_bootstrap 5  --target_layer_index 11
# python main_pretrain.py --num_bootstrap 8  --target_layer_index 11

# python main_pretrain.py --num_bootstrap 2 --use_new_feature_predictor --target_layer_index 5
# python main_pretrain.py --num_bootstrap 4 --use_new_feature_predictor --target_layer_index 5
# python main_pretrain.py --num_bootstrap 5 --use_new_feature_predictor --target_layer_index 5
# python main_pretrain.py --num_bootstrap 8 --use_new_feature_predictor --target_layer_index 5

# python main_pretrain.py --num_bootstrap 2 --use_new_feature_predictor --target_layer_index 11
# python main_pretrain.py --num_bootstrap 4 --use_new_feature_predictor --target_layer_index 11
# python main_pretrain.py --num_bootstrap 5 --use_new_feature_predictor --target_layer_index 11
# python main_pretrain.py --num_bootstrap 8 --use_new_feature_predictor --target_layer_index 11

# python main_pretrain.py --enable_ema --target_layer_index 5 --ema_decay 0.9999
# python main_pretrain.py --enable_ema --target_layer_index 5 --ema_decay 0.999
# python main_pretrain.py --enable_ema --target_layer_index 11 --ema_decay 0.99
# python main_pretrain.py --enable_ema --target_layer_index 11 --ema_decay 0.9

python main_pretrain.py --enable_ema --target_layer_index 11 --ema_decay 0.9999 --batch_size 512
python main_pretrain.py --enable_ema --target_layer_index 11 --ema_decay 0.999 --batch_size 512
# python main_pretrain.py --enable_ema --target_layer_index 11 --ema_decay 0.99 --batch_size 512
# python main_pretrain.py --enable_ema --target_layer_index 11 --ema_decay 0.9 --batch_size 512

python main_pretrain.py --enable_ema --target_layer_index 5 --ema_decay 0.9999 --batch_size 512
python main_pretrain.py --enable_ema --target_layer_index 5 --ema_decay 0.999 --batch_size 512
python main_pretrain.py --enable_ema --target_layer_index 5 --ema_decay 0.99 --batch_size 512
python main_pretrain.py --enable_ema --target_layer_index 5 --ema_decay 0.9 --batch_size 512

# iterate over the 'checkpoints/pretrain' directory
# for file in checkpoints/pretrain/*.pth; do
#     python main_linprobe.py --finetune $file
#     python main_finetune.py --finetune $file
# done