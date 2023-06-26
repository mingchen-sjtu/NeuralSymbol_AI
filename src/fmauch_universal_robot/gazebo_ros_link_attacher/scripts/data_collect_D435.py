import pyrealsense2 as rs
import cv2
import os
import numpy as np
import png
import glob


def return_max(path):
    files = []
    for file in glob.glob(path):
        files.append(os.path.basename(file))
    if not files:
        max_num = 0
    else:
        files.sort(key=lambda x: int(x[:-4]))
        str = files[-1]
        max_num = int(str.split('.', 1)[0])
    return max_num


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

pipeline_profile = pipeline.start(config)
depth_sensor = pipeline_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
depth_sensor.set_option(rs.option.visual_preset, 5)  # 5 对应short_range

align_to = rs.stream.color
align = rs.align(align_to)
colorizer = rs.colorizer()
i = return_max('D:\\Realsense\\HandEye_in_hand\\TCP0\\*.jpg')

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    # aligned_depth_frames = aligned_frames.get_depth_frame()
    aligned_color_frame = aligned_frames.get_color_frame()
    # if not aligned_depth_frames or not aligned_color_frame:
    #     continue
    # depth_img = np.asanyarray(colorizer.colorize(aligned_depth_frames).get_data())
    # depth_img_scale = cv2.resize(depth_img, (960, 540))
    # depth_data = np.asanyarray(aligned_depth_frames.get_data())

    color_img = np.asanyarray(aligned_color_frame.get_data())

    # depth_img = np.asanyarray(aligned_depth_frames.get_data())
    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
    # depth_img = depth_img * depth_scale
    # change the channels to RGB mode
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    # color_img_scale = cv2.resize(color_img, (960, 540))
    color_img_scale = cv2.resize(color_img, (640, 480))

    # cv2.namedWindow('depth_img', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('depth_img', depth_img_scale)
    # print("Depth:", depth_img)
    # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', color_img_scale)
    k = cv2.waitKey(1)
    # Esc退出，
    if k == 27:
        cv2.destroyAllWindows()
        break
    # 输入空格保存图片
    elif k == ord(' '):
        i = i + 1
        cv2.imwrite(os.path.join("/home/xps/Desktop/yolo/bolt_data_hole", str(i) + '.jpg'), color_img)
        print("RGB Frames{} Captured".format(i))
        # with open('D:\\Realsense\\pic_capture\\' + str(i) + "_d.jpg", 'wb') as f:
        #     writer = png.Writer(width=depth_data.shape[1], height=depth_data.shape[0],
        #                         bitdepth=16, greyscale=True)
        #     zgray2list = depth_data.tolist()
        #     writer.write(f, zgray2list)
        # print("Depth Frames{} Captured".format(i))
        # np.save(os.path.join("D:\\Realsense\\pic_capture\\not_aligned_with_obstacles", str(i) + '_d'), depth_data)
        # print("Depth Frames{} Captured".format(i))
pipeline.stop()
