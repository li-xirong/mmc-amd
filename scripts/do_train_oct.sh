cd code/
device=3 #Choose a GPU

run_id=1
train_collection="VisualSearch/mmc-amd-splitA-train"
val_collection="VisualSearch/mmc-amd-splitA-val"
configs_name="config-oct.py"
num_workers=4

python train-single.py --train_collection $train_collection \
                --val_collection $val_collection \
                --model_configs $configs_name \
                --run_id $run_id \
                --device $device \
                --num_workers $num_workers \
                --overwrite