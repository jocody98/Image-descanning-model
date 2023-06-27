import os
import csv
import cv2
import numpy as np

folder_path = './output'
output_file = 'output.csv'

# 폴더 내 이미지 파일 이름 목록을 가져오기
file_names = os.listdir(folder_path)
file_names.sort()

# CSV 파일을 작성하기 위해 오픈
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image File', 'Y Channel Value'])

    for file_name in file_names:
        # 이미지 로드
        image_path = os.path.join(folder_path, file_name)

        image = cv2.imread(image_path)

        # 이미지를 YUV 색 공간으로 변환
        image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # Y 채널 추출
        y_channel = image_yuv[:, :, 0]

        # Y 채널을 1차원 배열로 변환
        y_values = np.mean(y_channel.flatten())

        # print(y_values)

        # 파일 이름과 Y 채널 값을 CSV 파일에 작성
        writer.writerow([file_name[:-4], y_values])



# folder_path = './dataset/test/clean'
# output_file = 'real.csv'

# # 폴더 내 이미지 파일 이름 목록을 가져오기
# file_names = os.listdir(folder_path)
# file_names.sort()

# # CSV 파일을 작성하기 위해 오픈
# with open(output_file, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Image File', 'Y Channel Value'])

#     for file_name in file_names:
#         # 이미지 로드
#         image_path = os.path.join(folder_path, file_name)

#         image = cv2.imread(image_path)

#         # 이미지를 YUV 색 공간으로 변환
#         image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

#         # Y 채널 추출
#         y_channel = image_yuv[:, :, 0]

#         # Y 채널을 1차원 배열로 변환
#         y_values = np.mean(y_channel.flatten())

#         # print(y_values)

#         # 파일 이름과 Y 채널 값을 CSV 파일에 작성
#         writer.writerow([file_name[:-4], y_values])




csv_file = "real.csv"
reals = []

with open(csv_file, "r") as file:
    reader = csv.reader(file)
    next(reader) 

    for row in reader:
        value = float(row[-1]) 
        reals.append(value)

csv_file = "output.csv"
outputs = []

with open(csv_file, "r") as file:
    reader = csv.reader(file)
    next(reader) 

    for row in reader:
        value = float(row[-1])
        outputs.append(value)

mae = 0
for i in range(500):
    ae = abs(reals[i] - outputs[i])
    mae += ae
mae /= 500

print(mae)