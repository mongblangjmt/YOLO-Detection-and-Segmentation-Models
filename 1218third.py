import os
import cv2 as cv
from ultralytics import YOLO

def perform_detection_video(video_path, output_dir):
    # 사용할 모델 목록 (탐지와 분할 포함)
    models = [
        ("yolov8n.pt", "Detection"),  # yolov8n: 경량화된 탐지 모델
        ("yolov8n-seg.pt", "Segmentation"),  # yolov8n-seg: 분할 모델
        ("yolov8m.pt", "Detection"),  # yolov8m: 고성능 탐지 모델
        ("yolov8m-seg.pt", "Segmentation")  # yolov8m-seg: 고성능 분할 모델
    ]

    # Confidence thresholds
    conf_thresholds = [0.2, 0.5, 0.8]

    # 비디오 속성 가져오기
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return

    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))

    # 각 모델에 대해 탐지 수행
    for model_path, task in models:
        model = YOLO(model_path)
        for conf in conf_thresholds:
            # 출력 비디오 초기화
            output_video_path = os.path.join(output_dir, f"{os.path.basename(video_path).split('.')[0]}_{model_path.split('.')[0]}_conf{int(conf*100)}.mp4")
            fourcc = cv.VideoWriter_fourcc(*'mp4v')  # MP4 코덱
            out = cv.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            # 비디오 재생 초기화
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 탐지 수행
                results = model(frame, conf=conf)

                # 탐지된 객체와 세그먼트 표시
                annotated_frame = results[0].plot()  # 객체와 세그먼트가 포함된 프레임 생성

                # 결과 프레임 저장
                out.write(annotated_frame)

                # 결과 프레임 표시 (선택 사항)
                cv.imshow(f'Object Detection - {model_path} - Conf: {conf}', annotated_frame)
                if cv.waitKey(1) == ord('q'):
                    break

            print(f"Saved video {output_video_path}")
            out.release()

    cap.release()
    cv.destroyAllWindows()

# 실행 파라미터 설정
video_path = r"C:\Users\USER\OneDrive\바탕 화면\dlsrhdwlsmdrlakf\homevideo.mp4"  # 분석할 비디오 경로
output_dir = r"C:\Users\USER\OneDrive\바탕 화면\dlsrhdwlsmdrlakf\results"  # 결과 저장 디렉토리

# 탐지 수행
perform_detection_video(video_path, output_dir)
