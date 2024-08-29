import cv2
import torch
import numpy as np
from pathlib import Path
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

# 모델 로드
weights_path = Path('C:/research_2024/아니/yolov5/best.pt')  # Path 객체를 사용하여 경로 처리
model = attempt_load(weights_path)
model.eval()

# 웹캠 캡처 설정
VidioSignal = cv2.VideoCapture(0)
VidioSignal.set(cv2.CAP_PROP_FPS, 120)  # 프레임 속도를 30 FPS로 설정
cv2.namedWindow('YOLOv5_CM_01')

while True:
    ret, frame = VidioSignal.read()
    if not ret:
        break

    # 이미지 전처리
    img = letterbox(frame, new_shape=640)[0]  # 리사이즈
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    img = np.ascontiguousarray(img)  # contiguous array로 변환
    img = torch.from_numpy(img).float()  # numpy array -> torch tensor
    img /= 255.0  # 정규화
    img = img.unsqueeze(0)  # 배치 차원 추가

    # 모델에 이미지 전달 및 예측 수행
    with torch.no_grad():
        pred = model(img)[0]

    # NMS 적용
    pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.4)

    # 결과 시각화
    for det in pred:
        if len(det):
            det[:, :4] = det[:, :4]  # Bounding box xyxy
            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]} {conf:.2f}'
                x1, y1, x2, y2 = [int(x) for x in xyxy]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 결과 화면에 표시
    cv2.imshow('YOLOv5_CM_01', frame)
    cv2.resizeWindow('YOLOv5_CM_01', 650, 500)

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
VidioSignal.release()
cv2.destroyAllWindows()
