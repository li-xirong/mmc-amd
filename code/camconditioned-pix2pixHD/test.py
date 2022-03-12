### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import tqdm
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
import torch
from PIL import Image

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.gan_aug = False
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

if not os.path.exists(opt.synthesis_dst):
    os.mkdir(opt.synthesis_dst)

# test
model = create_model(opt)
if opt.data_type == 16:
    model.half()
elif opt.data_type == 8:
    model.type(torch.uint8)

if opt.verbose:
    print(model)
    
for i, data in enumerate(tqdm.tqdm(dataset)):
    if i >= opt.how_many:
        pass
    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['inst'] = data['inst'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['inst'] = data['inst'].uint8()
    minibatch = 1

    generated = model.inference(data['label'], data['inst'], data['image'])
    img_path = data['path']
    imgname = os.path.split(img_path[0])[-1][:-3]+"jpg"
    image_numpy = util.tensor2im(generated.data[0])
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(os.path.join(opt.synthesis_dst, imgname))
