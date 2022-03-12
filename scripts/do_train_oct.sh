cd code/
device=3 #Choose a GPU
for ((i=4;i<=5;i++))
do
run_id=$i
train_collection="VisualSearch/mmc-amd-splitP-train"
val_collection="VisualSearch/mmc-amd-splitP-val"
test_collection="VisualSearch/mmc-amd-splitP-test"
configs_name="config-oct.py"
num_workers=4

python train-single.py --train_collection $train_collection \
                --val_collection $val_collection \
                --test_collection $test_collection \
                --model_configs $configs_name \
                --run_id $run_id \
                --device $device \
                --num_workers $num_workers \
                --overwrite
done
