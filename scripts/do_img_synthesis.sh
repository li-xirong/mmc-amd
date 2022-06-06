if [ "$#" -ne 1 ]; then
    echo "Usage: $0 modality # cfp or oct"
    exit
fi
cd code/camconditioned_pix2pixHD/

modality=$1
data_root="datasets/"$modality"/"
synthesis_dir="synthesis_"$modality"/"
split_name="splitA"
gpu_ids="2"

python test.py --camlabel --name $split_name"_"$modality"_512" --dataroot $data_root \
               --synthesis_dst $synthesis_dir --netG local --ngf 32 \
               --resize_or_crop resize_and_crop --loadSize 512 --fineSize 512 \
               --no_instance --gpu_ids $gpu_ids