import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# 1. ページの基本設定
st.set_page_config(page_title="動画物体検知・追跡アプリ", layout="centered")
st.title("🎬 動画対応！車・自転車の検知＆追跡アプリ")
st.write("動画ファイルをアップロードすると、AIがリアルタイムに検知・追跡（トラッキング）を行います。")

# 2. YOLOv8モデルの読み込み
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")

model = load_model()

# 3. サイドバーの作成（前回までの機能を維持）
st.sidebar.header("🛠️ 検知設定")

st.sidebar.subheader("🎯 検知の厳しさ (Confidence)")
conf_threshold = st.sidebar.slider(
    "確信度のしきい値",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)

st.sidebar.subheader("📦 検知対象の選択")
detect_car = st.sidebar.checkbox("🚗 自動車（トラック含む）", value=True)
detect_bicycle = st.sidebar.checkbox("🚲 自転車", value=True)

selected_classes = []
if detect_car:
    selected_classes.extend([2, 3, 7])
if detect_bicycle:
    selected_classes.append(1)

# 4. 動画ファイルアップローダーの配置
# 【修正】typeを「mp4」に変更
uploaded_video = st.file_uploader(
    "動画ファイル（mp4）をアップロードしてください...", type=["mp4"]
)

# 動画がアップロードされた場合の処理
if uploaded_video is not None:
    
    # 選択されていない場合の警告
    if not selected_classes:
        st.sidebar.warning("⚠️ 検知対象を1つ以上選択してください。")
    else:
        # --- 動画の読み込み準備 ---
        # Streamlitが受け取った動画データを、Pythonが読み込めるように一時ファイルとして保存します
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        # OpenCVで動画を開く
        cap = cv2.VideoCapture(tfile.name)
        
        # 画面にパラパラ漫画を映し出すための「空の枠」を用意
        frame_placeholder = st.empty()
        
        # 処理中のメッセージ
        st.info("🔄 動画を解析して再生中...（途中で止める場合はブラウザの「Stop」ボタンを押してください）")

        # --- ここから追加：リアルタイムカウンターの表示枠を先に作成 ---
        st.subheader("📊 現在のリアルタイム検知状況")
        # 画面を2つの列に分け、自動車用と自転車用のカウンターを横並びにする
        col1, col2 = st.columns(2)
        car_metric = col1.metric(label="🚗 自動車 (画面内)", value=0)
        bike_metric = col2.metric(label="🚲 自転車 (画面内)", value=0)
        # ---------------------------------------------------------
        # 動画の全フレームをループ処理
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                     break  # 動画が終了したらループを抜ける
            # YOLOv8の「追跡機能（track）」を使用
            results = model.track(
                source=frame, 
                classes=selected_classes, 
                conf=conf_threshold, 
                persist=True,
                verbose=False # ターミナルへの大量のログ出力を非表示にする
                )
            result = results[0]
            # --- ここから追加：現在のフレーム内にいる「車種別の台数」をカウント ---
            current_car_count = 0
            current_bike_count = 0
                            
            # 検知結果（result.boxes）が存在する場合のみカウント処理を行う
            if result.boxes is not None and len(result.boxes) > 0:
                # 画面内のすべての検知オブジェクトのクラスIDを取得
                # YOLOv8では、クラスIDは浮動小数点数(float)のTensorで返ってくることがあるため、int型に変換します
                class_ids = result.boxes.cls.int().cpu().tolist()
        
                # YOLOv8の標準モデル(COCOデータセット)のクラスID: 
                # 2 = car（自動車）, 7 = truck（トラック）, 5 = bus（バス）, 1 = bicycle（自転車）
                # ※もし「自動車」にトラックやバスも含める場合は、7や5もカウント対象にします
                for cid in class_ids:
                    if cid in [2, 5, 7]:  # 自動車・バス・トラック
                        current_car_count += 1
                    elif cid == 1:       # 自転車
                        current_bike_count += 1

            # カウンターの数値をリアルタイムに更新（上書き）
            car_metric.metric(label="🚗 自動車 (画面内)", value=current_car_count)
            bike_metric.metric(label="🚲 自転車 (画面内)", value=current_bike_count)
            # -----------------------------------------------------------------

            # --- 結果の描画 ---
            # AIが枠線や追跡IDを描き込んだ画像を取得
            annotated_frame = result.plot()
            
            # OpenCVはBGR形式なので、Streamlitで表示するためにRGB形式に変換
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # 用意しておいた「空の枠」に現在のフレームの画像を上書き表示
            frame_placeholder.image(annotated_frame_rgb, use_container_width=True)

        # 使い終わった動画ファイルを閉じて、一時ファイルを削除
        cap.release()
        os.unlink(tfile.name)
        
        st.success("🎉 動画の解析が完了しました！")