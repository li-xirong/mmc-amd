cd code/
device=2
checkpoint="VisualSearch/mmc-amd-splitA-train/models/mmc-amd-splitA-val/config-cfp.py/run_1/best_epoch38_0.7702.pth"
checkpoint="VisualSearch/mmc-amd-splitA-train/models/mmc-amd-splitA-val/config-oct.py/run_2/best_epoch19_0.8913.pth"
collection="VisualSearch/mmc-amd-splitA-train"
configs_name="config-oct.py"
num_workers=4

python gencam.py --collection $collection --checkpoint $checkpoint\
  --model_configs $configs_name  \
  --device $device --num_workers $num_workers

