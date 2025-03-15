python main_pretrain.py --num_bootstrap 1 --target_layer_index 11 --model bmae_deit_tiny_patch4 

python main_pretrain.py --num_bootstrap 2 --target_layer_index 11 --model bmae_deit_tiny_patch4 
python main_pretrain.py --num_bootstrap 4 --target_layer_index 11 --model bmae_deit_tiny_patch4 
python main_pretrain.py --num_bootstrap 5 --target_layer_index 11 --model bmae_deit_tiny_patch4 
python main_pretrain.py --num_bootstrap 8 --target_layer_index 11 --model bmae_deit_tiny_patch4 
# python main_pretrain.py --num_bootstrap 10 --target_layer_index 11 --model bmae_deit_tiny_patch4 
# python main_pretrain.py --num_bootstrap 20 --target_layer_index 11 --model bmae_deit_tiny_patch4 

# Use new feature predictor to keep image reconstruction ability, these models are used to determine the evaluation range of K
python main_pretrain.py --num_bootstrap 2 --use_new_feature_predictor --target_layer_index 11 --model bmae_deit_tiny_patch4 
python main_pretrain.py --num_bootstrap 4 --use_new_feature_predictor --target_layer_index 11 --model bmae_deit_tiny_patch4 
python main_pretrain.py --num_bootstrap 5 --use_new_feature_predictor --target_layer_index 11 --model bmae_deit_tiny_patch4 
python main_pretrain.py --num_bootstrap 8 --use_new_feature_predictor --target_layer_index 11 --model bmae_deit_tiny_patch4 
python main_pretrain.py --num_bootstrap 10 --use_new_feature_predictor --target_layer_index 11 --model bmae_deit_tiny_patch4 
python main_pretrain.py --num_bootstrap 20 --use_new_feature_predictor --target_layer_index 11 --model bmae_deit_tiny_patch4 

python main_pretrain.py --num_bootstrap 2 --target_layer_index 5 --model bmae_deit_tiny_patch4 
python main_pretrain.py --num_bootstrap 4 --target_layer_index 5 --model bmae_deit_tiny_patch4 
python main_pretrain.py --num_bootstrap 5 --target_layer_index 5 --model bmae_deit_tiny_patch4 
python main_pretrain.py --num_bootstrap 8 --target_layer_index 5 --model bmae_deit_tiny_patch4 
# python main_pretrain.py --num_bootstrap 10 --target_layer_index 5 --model bmae_deit_tiny_patch4 
# python main_pretrain.py --num_bootstrap 20 --target_layer_index 5 --model bmae_deit_tiny_patch4 

python main_pretrain.py --num_bootstrap 2 --use_new_feature_predictor --target_layer_index 5 --model bmae_deit_tiny_patch4 
python main_pretrain.py --num_bootstrap 4 --use_new_feature_predictor --target_layer_index 5 --model bmae_deit_tiny_patch4 
python main_pretrain.py --num_bootstrap 5 --use_new_feature_predictor --target_layer_index 5 --model bmae_deit_tiny_patch4 
python main_pretrain.py --num_bootstrap 8 --use_new_feature_predictor --target_layer_index 5 --model bmae_deit_tiny_patch4 
# python main_pretrain.py --num_bootstrap 10 --use_new_feature_predictor --target_layer_index 5 --model bmae_deit_tiny_patch4 
# python main_pretrain.py --num_bootstrap 20 --use_new_feature_predictor --target_layer_index 5 --model bmae_deit_tiny_patch4 

# test hog feature
python main_pretrain.py --num_bootstrap 2 --feature_class hog --use_new_feature_predictor
python main_pretrain.py --num_bootstrap 4 --feature_class hog --use_new_feature_predictor

# EMA
python main_pretrain.py --enable_ema --ema_decay 0.9 --target_layer_index 11
python main_pretrain.py --enable_ema --ema_decay 0.99 --target_layer_index 11
python main_pretrain.py --enable_ema --ema_decay 0.999 --target_layer_index 11
python main_pretrain.py --enable_ema --ema_decay 0.9999 --target_layer_index 11

python main_pretrain.py --enable_ema --ema_decay 0.9 --target_layer_index 11 --use_new_feature_predictor
python main_pretrain.py --enable_ema --ema_decay 0.99 --target_layer_index 11 --use_new_feature_predictor
python main_pretrain.py --enable_ema --ema_decay 0.999 --target_layer_index 11 --use_new_feature_predictor
python main_pretrain.py --enable_ema --ema_decay 0.9999 --target_layer_index 11 --use_new_feature_predictor

python main_pretrain.py --enable_ema --target_layer_index 5 --ema_decay 0.9
python main_pretrain.py --enable_ema --target_layer_index 5 --ema_decay 0.99
python main_pretrain.py --enable_ema --target_layer_index 5 --ema_decay 0.999
python main_pretrain.py --enable_ema --target_layer_index 5 --ema_decay 0.9999

python main_pretrain.py --enable_ema --target_layer_index 5 --ema_decay 0.9999 --use_new_feature_predictor
python main_pretrain.py --enable_ema --target_layer_index 5 --ema_decay 0.999 --use_new_feature_predictor
python main_pretrain.py --enable_ema --target_layer_index 5 --ema_decay 0.99 --use_new_feature_predictor
python main_pretrain.py --enable_ema --target_layer_index 5 --ema_decay 0.9 --use_new_feature_predictor

# iterate over the 'checkpoints/pretrain' directory
for file in checkpoints/pretrain/*.pth; do
    python main_linprobe.py --finetune $file
    python main_finetune.py --finetune $file
done
