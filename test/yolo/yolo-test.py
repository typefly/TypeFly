import cv2
import torch
import numpy as np

def main():
    # load yolov5s model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    print("load image...")
    frame = cv2.imread("./kitchen.webp")

    # Detect objects
    results = model(frame)
    # row: xmin, ymin, xmax, ymax, confidence, class, name
    print(results.pandas().xyxy[0].values.tolist())
    for index, row in results.pandas().xyxy[0].iterrows():
        if row['confidence'] < 0.1:
            continue
        frame = cv2.putText(frame, row['name'], (int(row['xmin']), int(row['ymin']) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1, cv2.LINE_AA)
        frame = cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imwrite('result.jpg', frame)

if __name__ == '__main__':
    main()