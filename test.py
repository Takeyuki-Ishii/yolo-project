import cv2
import PIL
from ultralytics import YOLO

print("--- ライブラリの読み込み成功！ ---")

# YOLOv8の最も軽量なモデル（学習済みデータ）を自動ダウンロードして読み込み
model = YOLO("yolov8n.pt")
print("--- YOLOv8モデルの読み込み成功！ ---")