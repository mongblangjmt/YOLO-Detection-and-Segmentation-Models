import numpy as np
import cv2 as cv
from ultralytics import YOLO
import sys

def construct_yolo_v8(model_type='yolov8m.pt'):
    model = YOLO(model_type)  # 모델 로드
    class_names = model.names  # 클래스 이름들
    return model, class_names

def yolo_detect(img, yolo_model, conf_threshold=0.5):
    results = yolo_model(img)  # 객체 탐지
    objects = []

    # 탐지된 객체에서 신뢰도와 클래스 추출
    for result in results[0].boxes:
        x_center, y_center, width, height = result.xywh[0]
        confidence = result.conf[0]
        class_id = int(result.cls[0])

        x1, y1, x2, y2 = int(x_center - width / 2), int(y_center - height / 2), int(x_center + width / 2), int(y_center + height / 2)

        # 신뢰도가 임계값 이상일 경우만 객체로 인정
        if confidence > conf_threshold:
            objects.append([x1, y1, x2, y2, confidence, class_id])

    return objects

# 모델 리스트와 신뢰도 임계값 설정
models = ['yolov8m.pt', 'yolov8n.pt', 'yolov8n-seg.pt', 'yolov8m-seg.pt']
confidence_thresholds = [0.2, 0.5, 0.8]

# 입력 비디오 경로
input_video_path = 'C:/Users/USER/OneDrive/바탕 화면/dlsrhdwlsmdrlakf/homevideo.mp4'

# 비디오 속성 가져오기
cap = cv.VideoCapture(input_video_path)
if not cap.isOpened():
    sys.exit('동영상 파일을 열 수 없습니다.')

frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
video_duration = total_frames / fps

print(f"Input video properties: {frame_width}x{frame_height}, {fps} FPS, {video_duration:.2f} seconds")
cap.release()

# 모델과 신뢰도 조합별로 결과 저장
for model_type in models:
    model, class_names = construct_yolo_v8(model_type)
    colors = np.random.uniform(0, 255, size=(len(class_names), 3))  # 부류마다 색깔 생성

    print(f"Using model: {model_type}")

    for conf_threshold in confidence_thresholds:
        print(f"Confidence threshold: {conf_threshold}")

        # 비디오 재생 초기화
        cap = cv.VideoCapture(input_video_path)
        if not cap.isOpened():
            sys.exit(f'비디오 {input_video_path}를 다시 열 수 없습니다.')

        # 출력 비디오 초기화
        output_video_path = f"{model_type.split('.')[0]}_conf{int(conf_threshold*100)}.mp4"
        fourcc = cv.VideoWriter_fourcc(*'mp4v')  # MP4 코덱
        out = cv.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 객체 탐지
            objects = yolo_detect(frame, model, conf_threshold)

            # 탐지된 객체를 화면에 표시
            for obj in objects:
                x1, y1, x2, y2, confidence, id = obj
                text = f"{class_names[id]} {confidence:.3f}"
                cv.rectangle(frame, (x1, y1), (x2, y2), colors[id], 2)
                cv.putText(frame, text, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, colors[id], 2)

            # 결과 영상 저장
            out.write(frame)
            frame_count += 1

            # 결과 영상 표시
            cv.imshow(f'Object Detection - {model_type} - Conf: {conf_threshold}', frame)

            # 'q' 키를 누르면 종료
            key = cv.waitKey(1)
            if key == ord('q'):
                break

        print(f"Saved video {output_video_path} with {frame_count} frames.")
        out.release()
        cap.release()

cv.destroyAllWindows()

