import torch
from FCN_Train import FCNs
from FCN_Train import VGGNet
from CorridorInference import dataloader
import time
import visdom
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave

# dst_dir = './Datasets/CMU_corridor/predict/'
# dst_dir = './Datasets/UMichigan_corridor/Dataset_L_predict/'
# dst_dir = './Datasets/TQB_library_corridor/predict/'

# vis = visdom.Visdom()
fcn_model = torch.load('checkpoints/fcn_model_100.pth')
# fcn_model = fcn_model.cpu()
fcn_model.eval()
print('Done loading model!')

for item in dataloader:
    input = item['A']
    input = torch.autograd.Variable(input)
    input = input.cuda()

    input_np = input.cpu().data.numpy().copy()
    input_np = input_np[0]

    input_name = item['N'][0]
    input_name = input_name.split(".")[0]
    # print(input_name)

    output = fcn_model(input)
    output = torch.sigmoid(output)

    output_np = output.cpu().data.numpy().copy()
    output_np = np.argmin(output_np, axis=1)

    # imsave(dst_dir + input_name + '.png', output_np[0])

    # vis.close()
    # vis.images(input_np[:, None, :, :], opts=dict(title='raw'))
    # vis.images(output_np[:, None, :, :], opts=dict(title='pred'))

    fig = plt.figure('Inference')

    ax1 = fig.add_subplot(121)
    ax1.title.set_text('Raw' + input_name)
    ax1.imshow(np.squeeze(input_np[0, :, :]), 'gray')

    ax2 = fig.add_subplot(122)
    ax2.title.set_text('Predict' + input_name)
    ax2.imshow(np.squeeze(output_np[0, :, :]), 'gray')

    plt.pause(1)
