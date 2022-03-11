import os
import numpy as np
import cv2
from multiprocessing import pool
import ipdb
# 定义源视频根路径路径
video_root_Path = r"/public/home/dongsx/wsvad/data/shanghaitech/training/videos"
# 定义帧的存放路径
frame_root_path = "/public/home/dongsx/wsvad/data/shanghaitech/training/frames_bgr"
file_name = os.listdir(video_root_Path)
width=224
height =224
for f in file_name:  # 遍历video
    filename = os.path.splitext(f)[0]

    video_src_path = os.path.join(video_root_Path, f)

    frame_save_path = os.path.join(frame_root_path, filename)
    if not os.path.exists(frame_root_path):
        os.mkdir(frame_root_path)
    if not os.path.exists(frame_save_path):
        os.mkdir(frame_save_path)
    print(video_src_path)
    cap = cv2.VideoCapture(video_src_path)
    # frames = []
    i = 0
    if cap.isOpened():
        while True:
            success, frame_bgr = cap.read()
            if success is False:
                break

            # frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # frame_rgb = cv2.resize(frame_rgb, (width, height))
            # frames.append(frame_rgb)
            # cv2.imwrite(frame_rgb,)
            cv2.imwrite(frame_save_path + '/' + "{}.jpg".format(str(i).rjust(4, '0')), frame_bgr)
            i += 1
    cap.release()
    print('文件  ' + filename + '  转换完成 ')
    # try:
    #     print(video_src_path)
    #     cap = cv2.VideoCapture(video_src_path )
    #     # frames = []
    #     i = 0
    #     if cap.isOpened():
    #         while True:
    #             success, frame_bgr = cap.read()
    #             if success is False:
    #                 break
    #
    #             # frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    #             # frame_rgb = cv2.resize(frame_rgb, (width, height))
    #             # frames.append(frame_rgb)
    #             # cv2.imwrite(frame_rgb,)
    #             cv2.imwrite(frame_save_path + '/' + "{}.jpg".format(str(i).rjust(4,'0')), frame_bgr, params)
    #             i += 1
    #     cap.release()
    #     print('文件  ' + filename + '  转换完成 ')
    # except:
    #     print('error: ', filename, ' cannot open')
    #     raise error

    # forEach(videos, video_src_path, frame_save_path)


# np.save('F:/dataSet/meng_data/origin_video/label_dir.npy', label_dir)
print('程序运行完毕。。。')
