if [ "$#" -ne 1 ]; then
    echo "Usage: $0 modality # cfp or oct"
    exit
fi
cd code/camconditioned_pix2pixHD/

modality=$1
data_root="datasets/"$modality"/"
cams_dir=$data_root"train_A/"
dst_dir=$data_root"test_A/"
split_name="splitA"
gpu_ids="2"

python manipulate.py --cams_dir  $cams_dir \
                     --dst_dir  $dst_dir \
                     --modality $modality

python train.py --no_instance --camlabel --norm batch --dataroot $data_root --niter 100 --niter_decay 100 \
                --name $split_name"_"$modality"_256" --gpu_ids $gpu_ids --resize_or_crop resize_and_crop \
                --loadSize 272  --fineSize 256  --batchSize 2

python train.py --name $split_name"_"$modality"_512" --dataroot $data_root --netG local --ngf 32 \
       --num_D 2 --load_pretrain "checkpoints/"$split_name"_"$modality"_256/" --niter 50 --niter_decay 50 \
       --niter_fix_global 20 --resize_or_crop resize_and_crop --loadSize 520 --camlabel \
       --fineSize 512 --no_instance --gpu_ids $gpu_ids --batchSize 1