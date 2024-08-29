import cv2
import torch
import numpy as np
from pathlib import Path
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

# 모델 로드 (CPU에서 실행)
weights_path = Path(r'C:\research_2024\아니\yolov5_right_turn_detecter-master\yolov5_right_turn_detecter-master\best.pt')
model = attempt_load(weights_path)
model.eval()

# 동영상 파일 경로 설정
video_path = r'C:\research_2024\아니\yolov5_right_turn_detecter-master\yolov5_right_turn_detecter-master\20240811_184920.mp4'
VideoSignal = cv2.VideoCapture(video_path)

if not VideoSignal.isOpened():
    print("Error: Could not open video.")
    exit()

# 동영상 프레임 속도 및 크기 확인
fps = VideoSignal.get(cv2.CAP_PROP_FPS)
frame_width = int(VideoSignal.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(VideoSignal.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(VideoSignal.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video FPS: {fps}, Frame Size: {frame_width}x{frame_height}, Total Frames: {frame_count}")

# 트랙바 이벤트를 위한 글로벌 변수
current_frame = 0
frame_jump = False

# 트랙바 콜백 함수
def on_trackbar(val):
    global current_frame, frame_jump
    current_frame = val
    frame_jump = True
    VideoSignal.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

# 윈도우 창 생성 및 크기 설정
cv2.namedWindow('YOLOv5_CM_01', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLOv5_CM_01', frame_width, frame_height)

# 트랙바 생성
cv2.createTrackbar('Position', 'YOLOv5_CM_01', 0, frame_count - 1, on_trackbar)

while True:
    if not frame_jump:  # 트랙바로 위치를 이동하지 않은 경우에만 읽음
        ret, frame = VideoSignal.read()
        current_frame += 1
    else:
        ret = True
        frame_jump = False

    if not ret:
        break

    # 이미지 전처리
    img = letterbox(frame, new_shape=640)[0]  # 크기를 줄이면 처리 속도가 빨라질 수 있음
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    img /= 255.0
    img = img.unsqueeze(0)

    # 모델에 이미지 전달 및 예측 수행
    with torch.no_grad():
        pred = model(img)[0]

    # NMS 적용
    pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.4)

    # 결과 시각화
    for det in pred:
        if len(det):
            # 원본 프레임 크기에 맞게 좌표 스케일링
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]} {conf:.2f}'
                x1, y1, x2, y2 = [int(x) for x in xyxy]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 결과 화면에 표시
    cv2.imshow('YOLOv5_CM_01', frame)

    # 트랙바 업데이트
    cv2.setTrackbarPos('Position', 'YOLOv5_CM_01', current_frame)

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
VideoSignal.release()
cv2.destroyAllWindows()