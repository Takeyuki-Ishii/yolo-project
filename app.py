import os
import tempfile
import cv2
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
from ultralytics import YOLO

# --- 定数・制限値の設定 ---
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_VIDEO_DURATION_SEC = 30.0

# 1. 🎬 モデル・初期設定
st.set_page_config(page_title="動画物体検知・追跡アプリ", layout="centered")

@st.cache_resource
def load_model() -> YOLO:
    """YOLOv8モデルをキャッシュ読み込み"""
    return YOLO("yolov8s.pt")

model = load_model()

# 2. 🛡️ バリデーション関数(ファイルサイズ)
def validate_video_file(uploaded_file) -> bool:
    """
    アップロードされた動画ファイルのサイズを検証する。
    不適切な場合は st.error を表示して False を返す。
    """
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        st.error(f"❌ ファイルサイズが大きすぎます。ポートフォリオ環境保護のため、{MAX_FILE_SIZE_MB}MB以下の動画を選択してください。")
        return False
    return True

# 3. バリデーション関数(長さ)
def validate_video_duration(video_path: str) -> bool:
    """
    動画の長さを検証する。
    不適切な場合は st.error を表示して False を返す。
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    duration_sec = 0
    if fps > 0:
        duration_sec = total_frames / fps
        
    if duration_sec > MAX_VIDEO_DURATION_SEC:
        st.error(f"❌ 動画の長さが長すぎます（現在の動画: {duration_sec:.1f}秒）。サーバー負荷軽減のため、{MAX_VIDEO_DURATION_SEC}秒以内の動画を選択してください。")
        return False
    return True

# 4. 🧠 動画解析・YOLO推論ロジック
def process_video_tracking(video_path: str, selected_classes: list, conf_threshold: float, frame_skip: int, target_width: int):
    """
    OpenCVとYOLOを使用して動画の物体追跡を行い、リアルタイムに画面描画しながらログを収集する。
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or pd.isna(fps):
        fps = 30.0

    # UIのプレースホルダーとメトリクスの準備
    frame_placeholder = st.empty()
    st.subheader("📊 現在のリアルタイム検知状況")
    col1, col2 = st.columns(2)
    car_metric = col1.metric(label="🚗 自動車 (画面内)", value=0)
    bike_metric = col2.metric(label="🚲 自転車 (画面内)", value=0)

    temp_logs = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # フレーム間引き
        if frame_count % frame_skip != 0:
            continue
            
        current_time_sec = round(frame_count / fps, 2)

        # リサイズ処理
        h, w = frame.shape[:2]
        if w > target_width:
            target_height = int(h * (target_width / w))
            frame = cv2.resize(frame, (target_width, target_height))

        # YOLO推論
        results = model.track(
            source=frame, 
            classes=selected_classes, 
            conf=conf_threshold, 
            persist=True,
            verbose=False
        )
        result = results[0]

        current_car_count = 0
        current_bike_count = 0

        if result.boxes is not None and len(result.boxes) > 0:
            class_ids = result.boxes.cls.int().cpu().tolist()
            track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [-1] * len(class_ids)

            for cid, tid in zip(class_ids, track_ids):
                label = ""
                if cid in [2, 5, 7]:
                    current_car_count += 1
                    label = "car"
                elif cid == 1:
                    current_bike_count += 1
                    label = "bicycle"
                
                if label != "":
                    temp_logs.append({
                        "Frame": frame_count,
                        "Time(sec)": current_time_sec,
                        "Track_ID": tid,
                        "Class": label
                    })

        # メトリクスと映像のリアルタイム更新
        car_metric.metric(label="🚗 自動車 (画面内)", value=current_car_count)
        bike_metric.metric(label="🚲 自転車 (画面内)", value=current_bike_count)

        annotated_frame = result.plot()
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(annotated_frame_rgb, use_container_width=True)

    cap.release()
    return temp_logs

# 5. 📈 グラフ描画・結果表示UI
def display_tracking_dashboard(df_logs: pd.DataFrame):
    """
    解析完了後のユニーク台数集計、Altairグラフの描画、およびダウンロードUIを表示する。
    """
    st.success("🎉 動画の解析が完了しました！")
    
    # 固有の（ユニークな）台数を集計して表示
    total_unique_cars = df_logs[df_logs["Class"] == "car"]["Track_ID"].nunique() if "Class" in df_logs.columns else 0
    total_unique_bikes = df_logs[df_logs["Class"] == "bicycle"]["Track_ID"].nunique() if "Class" in df_logs.columns else 0
    
    st.write(f"🚗 検知された自動車の総数（ユニーク）: {total_unique_cars} 台")
    st.write(f"🚲 検知された自転車の総数（ユニーク）: {total_unique_bikes} 台")
    
    # 時系列グラフの作成
    st.subheader("⏱️ 時間経過ごとの検知台数の推移")
    
    if "Time(sec)" in df_logs.columns and "Track_ID" in df_logs.columns:
        df_filtered = df_logs[df_logs["Class"].isin(["car", "bicycle"])]
        
        if not df_filtered.empty:
            # グラフデータの集計ロジック
            df_frame_counts = df_filtered.groupby(["Time(sec)", "Class"])["Track_ID"].nunique().reset_index()
            df_frame_counts["Time_Rounded"] = (df_frame_counts["Time(sec)"].astype(float) / 1.0).round() * 1.0
            df_chart_data = df_frame_counts.groupby(["Time_Rounded", "Class"])["Track_ID"].max().astype(int).reset_index()
            df_chart_data.columns = ["Time(sec)", "Class", "Count"]
                                    
            # Altairチャートの構築
            color_scale = alt.Scale(domain=["car", "bicycle"], range=["#0000FF", "#FF0000"])
            chart = alt.Chart(df_chart_data).mark_line(strokeWidth=3).encode(
                x=alt.X("Time(sec):Q", title="経過時間 (秒)", axis=alt.Axis(grid=True)),
                y=alt.Y("Count:Q", title="検知台数 (台)", axis=alt.Axis(format="d")),
                color=alt.Color("Class:N", title="検知クラス", scale=color_scale),
                tooltip=[
                    alt.Tooltip("Time(sec):Q", title="時間(秒)"),
                    alt.Tooltip("Class:N", title="対象"),
                    alt.Tooltip("Count:Q", title="検知台数(台)")
                ]
            ).properties(width="container", height=350).interactive()
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("グラフを表示するための自動車・自転車のデータが不足しています。")
    else:
        st.error("ログデータに必要な列が存在しないため、グラフを描画できません。")
    
    # アクションボタン（ダウンロード、再解析）
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        csv = df_logs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="CSVをダウンロード",
            data=csv,
            file_name="tracking_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col2:
        if st.button("🔄 同じ動画で再解析する", use_container_width=True):
            st.session_state.analysis_done = False
            st.session_state.tracking_logs = []
            st.rerun()

# 6. 🛠️ メイン・アプリケーションフロー
def main():
    st.title("🎬 動画対応！車・自転車の検知＆追跡アプリ")
    st.write("動画ファイルをアップロードすると、AIがリアルタイムに検知・追跡（トラッキング）を行います。")

    # --- サイドバー設定 ---
    st.sidebar.header("🛠️ 検知設定")
    conf_threshold = st.sidebar.slider("確信度のしきい値", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
    
    st.sidebar.subheader("📦 検知対象の選択")
    detect_car = st.sidebar.checkbox("🚗 自動車（トラック含む）", value=True)
    detect_bicycle = st.sidebar.checkbox("🚲 自転車", value=True)

    selected_classes = []
    if detect_car: selected_classes.extend([2, 3, 7])
    if detect_bicycle: selected_classes.append(1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("パフォーマンス設定 (軽量化)")
    frame_skip = st.sidebar.slider("処理するフレーム間隔", min_value=1, max_value=6, value=3, step=1, help="推奨: 3")
    resolution_option = st.sidebar.selectbox("解析画質（動画の横幅）", options=["高画質 (そのまま)", "標準 (640px)", "軽量 (480px)"], index=1)

    target_width = 99999 if resolution_option == "高画質 (そのまま)" else (640 if resolution_option == "標準 (640px)" else 480)

    # --- ファイルアップローダー ---
    st.caption("⚠️ **【重要】制限事項：20MB以内、かつ30.0秒以内の動画のみ解析可能です。**")
    uploaded_video = st.file_uploader("動画ファイル（mp4）をアップロードしてください...", type=["mp4"])
    
    if uploaded_video:
        # 新しいファイルが上がったらセッション状態をリセット
        if "current_file" not in st.session_state or st.session_state.current_file != uploaded_video.name:
            st.session_state.current_file = uploaded_video.name
            st.session_state.analysis_done = False
            st.session_state.tracking_logs = []

    # セッション記憶の初期化
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "tracking_logs" not in st.session_state:
        st.session_state.tracking_logs = []

    # --- メインロジック分岐 ---
    if uploaded_video is not None:
        if not selected_classes:
            st.sidebar.warning("⚠️ 検知対象を1つ以上選択してください。")
            return

        # 1. ファイルサイズのバリデーション
        if not validate_video_file(uploaded_video):
            return

        # ファイルポインタを先頭に戻す（再解析時の空書き込み対策）
        uploaded_video.seek(0)

        # 一時ファイルへの保存処理
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        # 2. 動画の長さのバリデーション
        if not validate_video_duration(tfile.name):
            os.unlink(tfile.name)
            return

        # 3. 推論・解析フェーズ
        if not st.session_state.analysis_done:
            st.info("🔄 動画を解析して再生中...（途中で止める場合はブラウザの「Stop」ボタンを押してください）")
            
            # 追跡処理の実行
            logs = process_video_tracking(
                video_path=tfile.name,
                selected_classes=selected_classes,
                conf_threshold=conf_threshold,
                frame_skip=frame_skip,
                target_width=target_width
            )
            # 結果をセッションに格納して状態変更
            st.session_state.tracking_logs = logs
            st.session_state.analysis_done = True
            os.unlink(tfile.name)
            st.rerun()

        # 4. 結果表示・ダッシュボードフェーズ
        # ※「if not ...」と同じインデントレベル（関数 main の直下）に配置します
        if st.session_state.analysis_done:
            df_logs = pd.DataFrame(st.session_state.tracking_logs)
            
            if not df_logs.empty and "Class" in df_logs.columns:
                display_tracking_dashboard(df_logs)
            else:
                st.warning("⚠️ 動画内で物体は検知されなかったため、ダウンロードできるCSVデータはありません。")
                st.write("🚗 検知された自動車の総数: 0 台")
                st.write("🚲 検知された自転車の総数: 0 台")
                
                if st.button("🔄 設定を変えて再解析する", use_container_width=True):
                    st.session_state.analysis_done = False
                    st.session_state.tracking_logs = []
                    st.rerun()

# アプリケーションのエントリーポイント
# ※ 関数 main() の外側（インデントなし）に配置します
if __name__ == "__main__":
    main()