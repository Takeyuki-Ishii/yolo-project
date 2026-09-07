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

        # 動画の全フレームをループ処理
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # 動画が終了したらループを抜ける

            # 【重要】YOLOv8の「追跡機能（track）」を使用します！
            # model() ではなく model.track() を使うことで、フレーム間で同じ物体に同じIDを割り振ってくれます
            # persist=True で、前のフレームの記憶を引き継ぎます
            results = model.track(
                source=frame, 
                classes=selected_classes, 
                conf=conf_threshold, 
                persist=True,
                verbose=False # ターミナルへの大量のログ出力を非表示にする
            )
            
            result = results[0]

            # --- 結果の描画 ---
            # AIが枠線や追跡ID（予測された番号）を描き込んだ画像を取得
            annotated_frame = result.plot()
            
            # OpenCVはBGR形式なので、Streamlitで表示するためにRGB形式に変換
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # 用意しておいた「空の枠」に現在のフレームの画像を上書き表示（これで動画に見えます）
            frame_placeholder.image(annotated_frame_rgb, use_container_width=True)

        # 使い終わった動画ファイルを閉じて、一時ファイルを削除
        cap.release()
        os.unlink(tfile.name)
        
        st.success("🎉 動画の解析が完了しました！")