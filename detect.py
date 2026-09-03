import cv2
from ultralytics import YOLO

# 1. YOLOv8モデルの読み込み
model = YOLO("yolov8n.pt")

# 2. 画像ファイルの指定
image_path = "car.jpg"

# 3. 物体検出の実行（自転車:1、車:2 に絞り込み）
results = model(image_path, conf=0.25, classes=[1, 2])

# 4. 検出結果の確認とカウント（新しく追加した処理）
for result in results:
    # 検出されたすべての物体のクラスID（背番号）を取得
    # .boxes.cls に [2.0, 2.0] のような形でデータが入っています
    detected_classes = result.boxes.cls.tolist()
    
    # リストの要素数を len() で数えることで、合計台数を計算
    total_count = len(detected_classes)
    
    # 内訳を数える（リストの中に 2 が何個あるか、1 が何個あるか）
    car_count = detected_classes.count(2)
    bicycle_count = detected_classes.count(1)
    
    # ターミナルに分かりやすく結果を表示
    print("\n==============================")
    print(f"【AIの解析結果】")
    print(f"画面内の合計乗り物数: {total_count} 台")
    print(f"・自動車: {car_count} 台")
    print(f"・自転車: {bicycle_count} 台")
    print("==============================\n")
    
    # 画像の保存
    result.save(filename="result.jpg")

print("--- 処理がすべて完了しました！ ---")