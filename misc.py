import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.misc import imsave
import cv2

# np.set_printoptions(threshold=np.inf)  # print full result

os.getcwd()

# Name mapping for CMU-corridor dataset
# collection = 'Datasets/2-UMichigan-corridor/predicted/'
# for i, filename in enumerate(os.listdir(collection)):
#     name_split = filename.split(".")
#     name = name_split[0]
#     # print(name)
#     os.rename(collection + filename, collection + 'img-' + name + '.png')
#
# collection = './ground_truth/'
# for i, filename in enumerate(os.listdir(collection)):
#     name_split = filename.split(".")
#     name = name_split[0][6:10]
#     print(name)
#     os.rename('./ground_truth/' + filename, './ground_truth_rename/' + name + '.png')

# Name mapping for UMichigan-corridor dataset
# src_dir = 'Datasets/2-UMichigan-corridor/png/'
# num_img = 37
# start_index = 1133
# for i in range(num_img):                        # ex: 0
#     i_offset = i + start_index                  # ex: 998
#     i_offset_str = str(i_offset)                # ex: '998'
#     i_offset_str_fill = i_offset_str.zfill(4)  # ex: '0998'
#     num = i * 10                                # ex: 0
#     num_str = str(num)                          # ex: '0'
#     num_str_fill = num_str.zfill(4)             # ex: '0000'
#
#     os.rename(src_dir + num_str_fill + '.png', src_dir + i_offset_str_fill + '.png')

# Name mapping for TQB-library-corridor dataset
# src_dir = 'Datasets/3-TQB-library-corridor/imgs/'
# dst_dir = 'Datasets/3-TQB-library-corridor/imgs/'

# src_dir = 'Datasets/LibNew/'
# dst_dir = 'Datasets/LibNew/'
#
# for i, filename in enumerate(os.listdir(src_dir)):
#     os.rename(src_dir + filename, dst_dir + str(i) + '.jpg')

# Display RGB image using matplotlib
# img = plt.imread('Datasets/2-UMichigan-corridor/Dataset_L_ground_truth/0000.png')
# plt.imshow(img, 'gray')
# plt.show()

# Convert RGB to BW, after labelling using labelme tool
# src_dir = 'Datasets/3-TQB-library-corridor/this-label/'
# dst_dir = 'Datasets/3-TQB-library-corridor/this-label/'
# src_dir = 'Datasets/xyz/test/labelme/'
# dst_dir = 'Datasets/xyz/test/ground-truth/'
# for i, filename in enumerate(os.listdir(src_dir)):
#     img_HxWxD = plt.imread(src_dir + filename)
#     img_DxHxW = np.moveaxis(img_HxWxD, -1, 0)
#     img_HxW = img_DxHxW[0]
#     img_HxW[img_HxW > 0] = 1.0
#
#     imsave(dst_dir + filename, img_HxW)  # keep the filename unchanged

# src_dir = 'Datasets/4-Self-collected-corridor/ground-truth/'
# start_index = 1270
# for i, filename in enumerate(os.listdir(src_dir)):
#     name_split = filename.split(".")
#     name_split = name_split[0]
#     name_int = int(name_split)
#     name_int_offset = name_int + start_index
#     name_str_offset = str(name_int_offset)
#     name_str_offset_fill = name_str_offset.zfill(4)
#     os.rename(src_dir + filename, src_dir + name_str_offset_fill + '.png')


# Using this when there are blended objects after doing labelling using labelme tool
# src_dir = 'Datasets/4-Self-collected-corridor/ground-truth/'
# dst_dir = 'Datasets/4-Self-collected-corridor/ground-truth/'
# img_ground = plt.imread(src_dir + '194-1.png')
# img_not_ground = plt.imread(src_dir + '194-2.png')
# img_ground[img_not_ground == 1.0] = 0.0
# imsave(dst_dir + '194.png', img_ground)


# import cv2
# import os
#
# image_folder = 'Datasets/2-UMichigan-corridor/raw-image/'
# video_name = 'video.avi'
#
# images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape
#
# video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (width,height))
#
# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))
#
# cv2.destroyAllWindows()
# video.release()

# Load log files
# a = np.load('logs/log-VGG16-FCNs-CMU-96x96.npz')
# val_mIoU = a['arr_3']
# max_mIoU = max(val_mIoU)
# print(max_mIoU)

# dataset_dir = 'Datasets/CMU-corridor-train-validation/val/'
# img_name = os.listdir(dataset_dir + 'raw-image/')[0]
# imgA = cv2.imread(dataset_dir + 'raw-image/' + img_name)
# imgB = cv2.imread(dataset_dir + 'ground-truth/' + img_name, 0)
# imgB = cv2.resize(imgB, (96, 96))
# print(imgB.shape)
# print(imgB)
# print('---')
#
# # imgB = cv2.cvtColor(imgB, cv2.COLOR_GRAY2RGB)
#
# imgB = imgB[..., np.newaxis]
# print(imgB)

# Dung nho lam
# imgs = []
# num_img = 25
# start_index = 87
# for i in range(num_img):                        # ex: 0
#     i_offset = i + start_index                  # ex: 87
#     i_offset_str = str(i_offset)                # ex: '87'
#     filename = i_offset_str + '.png'            # ex: '87.png'
#     img = cv2.imread(filename)
#     # do something
#     imgs.append(img)

# Flip image horizontally
# src_dir = 'Datasets/xyz/ground-truth/'
# dst_dir = 'Datasets/xyz/ground-truth-flip/'
#
# num_img = 2
#
# start_index = 0
# for i in range(num_img):                        # ex: 0
#     i_offset = i + start_index                  # ex: 87
#     i_offset_str = str(i_offset)                # ex: '87'
#     i_offset_str_fill = i_offset_str.zfill(4)  # ex: '0087'
#     filename = i_offset_str_fill + '.png'            # ex: '0087.png'
#     img = cv2.imread(src_dir + filename)
#     flip_img = cv2.flip(img, 1)
#     flip_img = cv2.cvtColor(flip_img, cv2.COLOR_BGR2RGB)
#     imsave(dst_dir + i_offset_str_fill + '_flip.png', flip_img)

# Randomly pick images and split into train-val-test with following ratio:
# num_train = 1238
# num_val = 310
# num_test = 386

# wtf = 'raw-image'
# wtf = 'ground-truth'
# src_dir = 'Datasets/CMU-corridor-train-validation/' + wtf + '/'
# dst_dir = 'Datasets/CMU-corridor-train-validation/'
#
#
# for i, filename in enumerate(os.listdir(src_dir)):
#     if i < num_train:
#         img = cv2.imread(src_dir + filename)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         imsave(dst_dir + 'train/' + wtf + '/' + filename, img)
#     elif num_train <= i < (num_train + num_val):
#         img = cv2.imread(src_dir + filename)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         imsave(dst_dir + 'val/' + wtf + '/' + filename, img)
#     else:
#         img = cv2.imread(src_dir + filename)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         imsave(dst_dir + 'test/' + wtf + '/' + filename, img)

# Extract frames from video
# import cv2
# import time
# import os
#
#
# def video_to_frames(input_path, output_path):
#     """Function to extract frames from input video file
#     and save them as separate frames in an output directory.
#     Args:
#         input_path: Input video file.
#         output_path: Output directory to save the frames.
#     Returns:
#         None
#     """
#     try:
#         os.mkdir(output_path)
#     except OSError:
#         pass
#     # Log the time
#     time_start = time.time()
#     # Start capturing the feed
#     cap = cv2.VideoCapture(input_path)
#     # Find the number of frames
#     video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
#     print("Number of frames: ", video_length)
#     count = 0
#     print("Converting video..\n")
#     # Start converting the video
#     while cap.isOpened():
#         # Extract the frame
#         ret, frame = cap.read()
#         # Write the results back to output location.
#         cv2.imwrite(output_path + "/%#05d.jpg" % (count + 1), frame)
#         count = count + 1
#         # If there are no more frames left
#         if (count > (video_length - 1)):
#             # Log the time again
#             time_end = time.time()
#             # Release the feed
#             cap.release()
#             # Print stats
#             print("Done extracting frames.\n%d frames extracted" % count)
#             print("It took %d seconds forconversion." % (time_end - time_start))
#             break
#
#
# input_path = '/home/nhan/Downloads/video.mp4'
# output_path = '/home/nhan/Downloads/frames/'
# video_to_frames(input_path, output_path)

# Overlay prediction

# mask = np.zeros((10, 10))
# mask[3:-3, 3:-3] = 1  # white square in black background
# im = mask + np.random.randn(10, 10) * 0.01  # random image
# masked = np.ma.masked_where(mask == 0, mask)
#
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(im, 'gray', interpolation='none')
# plt.subplot(1, 2, 2)
# plt.imshow(im, 'gray', interpolation='none')
# plt.imshow(masked, 'jet', interpolation='none', alpha=0.1)
# plt.show()

src_dir = 'Datasets/xyz/test/raw-image-96x96/'
pred_dir = 'Datasets/xyz/test/prediction/'
dst_dir = 'Datasets/xyz/test/prediction-overlay/'
for i, filename in enumerate(os.listdir(src_dir)):
    img = plt.imread(src_dir + filename)
    overlay = plt.imread(pred_dir + filename)
    masked = np.ma.masked_where(overlay == 0, overlay)

    plt.figure()

    plt.imshow(img)
    plt.imshow(masked, 'winter', alpha=0.2)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)

    # plt.show()
    plt.savefig(dst_dir + filename)

