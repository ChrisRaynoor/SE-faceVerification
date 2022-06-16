# ./settings.py
# 配置文件，包含全局设置

DB_ADDRESS = './database.sqlite'
FACE_VER_THRESHOLD = 0.8
FACE_VER_LONGEST_WAIT_S = 8 # 通过前的最长等待时间s
FACE_VER_REQUIRED_ACC = 5   # 需要的连续通过数量
FACENET_INPUT_IMAGE_SIZE = 160
FACENET_INPUT_MARGIN = 0
CAM_DISPLAY_SIZE = (640, 480)
CAM_CROPPED_DISPLAY_SIZE = (240, 320)
CAM_CROPPED_DISPLAY_LINE_RGB = (51, 222, 255) # 黄色