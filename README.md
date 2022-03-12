# MMC-AMD

##### |[Paper](https://arxiv.org/pdf/2012.01879)|

Code and data for multi-modal categorization of age-related macular degeneration (4 classes: normal, dry AMD, pcv, wet AMD)

<center>
    <img style="border-radius: 0.3125em;    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     src="images/pipeline.jpg">
    <br>
    <div style="color:orange;  display: inline-block;    color: black;    padding: 2px;" align="center"><h>Fig.1. PIPLINE</h></div>
</center>

## Download
* <b>Dataset</b>

    Data is freely available upon request and for ***research purposes*** only. Please submit your request via [Google Form](https://forms.gle/jJT6H9N9CY34gFBWA).

* <b>Pre-trained Models</b>
  | Model       | F1 score |  |
  | :---------: | :----: | :---- |
  | [CFP-CNN]() | ? | A resnet18 trained on color fundus images |
  | [OCT-CNN]() | ? | A resnet18 trained on OCT images |
  | [MM-CNN]()  | ? | A two-stream CNN trained on muilti-modal data with loose pair training and CAM-conditioned image synthesis |
Note that we pre-process color fundus images by CLAHE and oct by median blur, and then resize both to 448x448.

Please download the pre-trained weights above, and put them into ```./code/weights/```

## Requirements
* <b>Python-3.7.10</b>
* <b>CUDA-10.1</b>
* <b>Pytorch-1.1.0</b> & <b>torchvision-0.3.0</b>
```
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0
```
* <b>Other Packages</b>
```
pip install -r requirements.txt
```

## MMC-AMD Inference
#### We have prepared two jupyter notebook files for single-modal and multi-modal inference, respectively:  
```./code/inference-and-eval-single-modal.ipynb```

```./code/inference-and-eval-multi-modal.ipynb```

## MMC-AMD Training
#### 1. To train a color fundus singe-modal model, please run 
```bash scripts/do_train_cfp.sh```
#### 2. To train a color fundus singe-modal model, please run
```bash scripts/do_train_oct.sh```
#### 3. To train a multi-modal model without loose pair training, please run
```bash scripts/do_train_mm.sh```
#### 4. To train a multi-modal model with loose pair training, please run
```bash scripts/do_train_mm_loose.sh```

## CAM-conditioned image synthesis
#### 1. Prepare CAM-conditioned label
* Make sure you have a trained CFP-CNN and a trained OCT-CNN. 
* Run the command below to generate CFP CAMs and OCT CAMs, respectively

```bash scripts/do_generatecam.sh``` 
* link the CAM dir generated in the previous step to ```code/camconditioned-pix2pixHD/datasets/$DATASET_NAME/train_A```
* link the image dir (```code/VisualSearch/mmc-amd/ImageData/$MODALITY```) generated in the previous step to ```code/camconditioned-pix2pixHD/datasets/$DATASET_NAME/train_B```
#### 2. Train pix2pixHD and synthesize  
```bash scripts/do_synthesis_cfp.sh```
```bash scripts/do_synthesis_oct.sh```


### II. Train
#### 1. Data
Experimental dataset is made by synthetic images. The dataset are save as below: 

```
./code/VisualSearch/train/ (./code/VisualSearch/val/)
	annotations/
		fundus.txt
		oct.txt
	ImageSets/
		fundus.txt
		oct.txt
	fundus-images/
		f-*.jpg
	oct-imges/
		o-*.jpg
```
Note that we provide 10 synthetic images per class as a toy dataset. 


## Citations

If you find this repository useful, please consider citing:
```
@inproceedings{miccai19-mmcamd,
  author    = {Weisen Wang and Zhiyan Xu and Weihong Yu and Jianchun Zhao and Jingyuan Yang and Feng He and Zhikun Yang and Di Chen and Dayong Ding and Youxin Chen and Xirong Li},
  title     = {Two-Stream {CNN} with Loose Pair Training for Multi-modal {AMD} Categorization},
  booktitle = {MICCAI},
  pages     = {156--164},
  doi = {10.1007/978-3-030-32239-7_18},
  year      = {2019},
}

@article{arxiv-mmcamd,
  author={Weisen Wang and Xirong Li and Zhiyan Xu and Weihong Yu and Jianchun Zhao and Dayong Ding and Youxin Chen},
  title={Learning Two-Stream {CNN} for Multi-Modal Age-related Macular Degeneration Categorization},
  journal={arXiv preprint arXiv:2012.01879},
  doi={10.48550/arXiv.2012.01879},
  year={2020},
}
```

## Acknowledgments

* The code of CAM-conditioned pix2pixHD borrows from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
