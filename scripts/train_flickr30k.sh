SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

DEVICE=$1
EXP_NAME=`echo "$(basename $0)" | cut -d'.' -f1` 
LOG_FILE=logs/$EXP_NAME

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/${EXP_NAME}
LOG_FILE="$LOG_FOLDER/${TIME_START}.log"
mkdir -p $LOG_FOLDER

echo "=========================================================="
echo "RUNNING EXPERIMENTS: $EXP_NAME, saving in checkpoints/$EXP_NAME"
echo "=========================================================="

# Step 0 (once): python ilr/build_ilr_neighbors.py \
#   --path_of_datasets ./annotations/flickr30k/flickr30k_texts_features_ViT-B32.pickle \
#   --output_path ./annotations/flickr30k/flickr30k_ilr_neighbors_k5_seed30_var0.04.json

python main.py \
--bs 80 \
--lr 0.00002 \
--epochs 30 \
--device cuda:$DEVICE \
--random_mask \
--prob_of_random_mask 0.4 \
--clip_model ViT-B/32 \
--using_clip_features \
--language_model gpt2 \
--using_hard_prompt \
--soft_prompt_first \
--use_ilr \
--ilr_k 5 \
--ilr_neighbors_path ./annotations/flickr30k/flickr30k_ilr_neighbors_k5_seed30_var0.04.json \
--fusion_w1 0.85 \
--fusion_w2 0.15 \
--path_of_datasets ./annotations/flickr30k/flickr30k_texts_features_ViT-B32.pickle \
--out_dir checkpoints/$EXP_NAME \
--use_amp \
|& tee -a  ${LOG_FILE}