cd code/
device=2 #Choose a GPU

run_id=1
train_collection="VisualSearch/mmc-amd-splitA-train"
val_collection="VisualSearch/mmc-amd-splitA-val"
configs_name="config-mm-finetune.py"
num_workers=4
checkpoint="path_of_one_checkpoint_from_the_pretrain_step"

python train.py --train_collection $train_collection \
                --val_collection $val_collection \
                --model_configs $configs_name \
                --run_id $run_id \
                --device $device \
                --num_workers $num_workers \
                --overwrite \
                --checkpoint $checkpoint