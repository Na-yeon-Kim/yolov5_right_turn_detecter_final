import cv2
import torch
import numpy as np
from pathlib import Path
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, xyxy2xywh, box_iou
import pathlib
from pathlib import Path
import os  # 폴더 생성에 필요한 모듈
pathlib.PosixPath = pathlib.WindowsPath

# 모델 로드
weights_path = Path('C:/research_2024/아니/yolov5/best.pt')  # Path 객체를 사용하여 경로 처리
model = attempt_load(weights_path)
model.eval()

# 웹캠 캡처 설정
VideoSignal = cv2.VideoCapture(0)
VideoSignal.set(cv2.CAP_PROP_FPS, 120)  # 프레임 속도를 120 FPS로 설정
cv2.namedWindow('YOLOv5_CM_01')

# 이미지 저장을 위한 카운터 초기화
save_counter = 0
previous_detections = []

# licence_plate 폴더 생성
output_dir = Path('C:/research_2024/아니/yolov5/licence_plate')
output_dir.mkdir(parents=True, exist_ok=True)  # 폴더가 없으면 생성

def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes."""
    return box_iou(torch.tensor(box1).unsqueeze(0), torch.tensor(box2).unsqueeze(0)).item()

while True:
    ret, frame = VideoSignal.read()
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

    # 결과 시각화 및 감지된 licence_plate 저장
    for det in pred:
        if len(det):
            det[:, :4] = det[:, :4]  # Bounding box xyxy
            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]} {conf:.2f}'
                x1, y1, x2, y2 = [int(x) for x in xyxy]
                current_box = [x1, y1, x2, y2]

                # licence_plate 클래스가 감지된 경우
                if model.names[int(cls)] == 'licence_plate':
                    is_duplicate = False
                    for previous_box in previous_detections:
                        iou = compute_iou(current_box, previous_box)
                        if iou > 0.5:  # IoU가 0.5 이상이면 동일한 번호판으로 간주
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        save_counter += 1
                        plate_img = frame[y1:y2, x1:x2]  # 감지된 부분 자르기
                        save_path = output_dir / f'detected_plate_{save_counter}.jpg'  # 폴더 내에 저장
                        cv2.imwrite(str(save_path), plate_img)
                        print(f"Saved detected licence plate to {save_path}")
                        previous_detections.append(current_box)

                # 감지 결과 시각화
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 결과 화면에 표시
    cv2.imshow('YOLOv5_CM_01', frame)
    cv2.resizeWindow('YOLOv5_CM_01', 650, 500)

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
VideoSignal.release()
cv2.destroyAllWindows()
