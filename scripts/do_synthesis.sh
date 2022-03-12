cd code/camconditioned-pix2pixHD/

modality="oct"
data_root="datasets/oct_A/"
cams_dir=$data_root"train_A/"
dst_dir=$data_root"test_A/"
synthesis_dir="synthesis_oct_A/"
split_name="splitA"
gpu_ids="2"

python manipulate.py --cams_dir  $cams_dir \
                     --dst_dir  $dst_dir \
                     --modality $modality

python train.py --no_instance --camlabel --norm batch --dataroot $data_root --niter 10 --niter_decay 10 \
                --name $split_name"_"$modality"_256" --gpu_ids $gpu_ids --resize_or_crop resize_and_crop \
                --loadSize 272  --fineSize 256  --batchSize 8

python train.py --name $split_name"_"$modality"_512" --dataroot $data_root --netG local --ngf 32 \
       --num_D 2 --load_pretrain "checkpoints/"$split_name"_"$modality"_256/" --niter 5 --niter_decay 5 \
       --niter_fix_global 20 --resize_or_crop resize_and_crop --loadSize 520 --camlabel \
       --fineSize 512 --no_instance --gpu_ids $gpu_ids --batchSize 2

python test.py --camlabel --name $split_name"_"$modality"_512" --dataroot $data_root \
               --synthesis_dst $synthesis_dir --netG local --ngf 32 \
               --resize_or_crop resize_and_crop --loadSize 512 --fineSize 512 \
               --no_instance --gpu_ids $gpu_ids