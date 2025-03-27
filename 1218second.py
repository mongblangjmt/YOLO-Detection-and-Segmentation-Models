import os
from ultralytics import YOLO

def perform_detection_single_image(image_path, output_dir):

    # 사용할 모델 목록 (탐지와 분할 포함)
    models = [
        ("yolov8n.pt", "Detection"),  # yolov8n: 경량화된 탐지 모델
        ("yolov8n-seg.pt", "Segmentation"),  # yolov8n-seg: 분할 모델
        ("yolov8m.pt", "Detection"),  # yolov8m: 고성능 탐지 모델
        ("yolov8m-seg.pt", "Segmentation")  # yolov8m-seg: 고성능 분할 모델
    ]

    # Confidence thresholds
    conf_thresholds = [0.2, 0.5, 0.8]

    # 각 모델에 대해 탐지 수행
    for model_path, task in models:
        model = YOLO(model_path)
        for conf in conf_thresholds:
            # 탐지 수행
            results = model(image_path, conf=conf)

            # 결과 출력 (탐지된 객체가 있는지 확인)
            print(f"Results for model: {model_path}, task: {task}, confidence: {conf}")
            print(results)  # 탐지된 객체 출력

            # 결과 화면에 표시
            if isinstance(results, list):
                for result in results:  # 리스트에 대해서는 각 항목에 대해 show 호출
                    result.show()  # 화면에 표시
            else:
                results.show()  # 단일 결과에 대해 show 호출

# 실행 파라미터 설정
image_path = r"C:\Users\USER\OneDrive\바탕 화면\dlsrhdwlsmdrlakf\room.jpg"  # 분석할 이미지 경로
output_dir = r"C:\Users\USER\OneDrive\바탕 화면\dlsrhdwlsmdrlakf\results"  # 결과 저장 디렉토리 (저장 안 함)

# 탐지 수행
perform_detection_single_image(image_path, output_dir)
