import os
import tempfile
import cv2
import numpy as np
import streamlit as st
import pandas as pd  # 📊 CSV化のために必要です。
import altair as alt
from ultralytics import YOLO
# --- 制限値の設定（ポートフォリオの仕様に合わせて自由に変更してください） ---
MAX_FILE_SIZE_MB = 20 # 読み込みファイルのサイズ制限(MB)
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024  # バイト単位に変換
MAX_VIDEO_DURATION_SEC = 30.0  # 読み込みファイルの長さ制限

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

# ✨ 【追加】パフォーマンス設定（クラウド・軽量化対策）
st.sidebar.markdown("---") # 区切り線を入れて見やすくします
st.sidebar.subheader("パフォーマンス設定 (軽量化)")

# ① フレーム間引きの設定（スライダー）
# ユーザーが直感的に選べるよう「何フレームごと」という表現にします
frame_skip = st.sidebar.slider(
    "処理するフレーム間隔",
    min_value=1,
    max_value=6,
    value=3,
    step=1,
    help="数値を大きくすると処理が非常に軽くなりますが、動画の動きが飛び飛びになります（推奨: 3）"
)

# ② 解析解像度の設定（セレクトボックス）
# 画質（横幅のピクセル数）を選択できるようにします
resolution_option = st.sidebar.selectbox(
    "解析画質（動画の横幅）",
    options=["高画質 (そのまま)", "標準 (640px)", "軽量 (480px)"],
    index=1, # デフォルトで「標準 (640px)」を選択した状態にする
    help="クラウド環境では「標準」または「軽量」を選択することで、メモリ不足によるクラッシュを防げます"
)

# セレクトボックスの文字列を、OpenCVで使う数値（TARGET_WIDTH）に変換する処理
if resolution_option == "高画質 (そのまま)":
    target_width = 99999  # リサイズ処理をスキップさせるための大きな値
elif resolution_option == "標準 (640px)":
    target_width = 640
else:
    target_width = 480

# 4. 動画ファイルアップローダーの配置
# アップローダーの直前に小さな文字で注釈を追加
st.caption("⚠️ **【重要】制限事項：20MB以内、かつ30.0秒以内の動画のみ解析可能です。**")
uploaded_video = st.file_uploader(
    "動画ファイル（mp4）をアップロードしてください...", 
    type=["mp4"],
    help="サーバー負荷軽減のため、サイズは20MBまで、長さは30.0秒までの制限を設けています。制限を超える動画はアップロード後にエラー表示となります。"
)
if uploaded_video:
    # 新しいファイル名がセッションに保存されているものと違う場合、または初めての場合にリセット
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_video.name:
        st.session_state.current_file = uploaded_video.name
        st.session_state.analysis_done = False
        st.session_state.tracking_logs = []

# 動画がアップロードされた場合の処理
if uploaded_video is not None:
    
    # 選択されていない場合の警告
    if not selected_classes:
        st.sidebar.warning("⚠️ 検知対象を1つ以上選択してください。")
    
    # 【追加】① ファイルサイズの事前チェック
    elif uploaded_video.size > MAX_FILE_SIZE_BYTES:
        st.error(f"❌ ファイルサイズが大きすぎます。ポートフォリオ環境保護のため、{MAX_FILE_SIZE_MB}MB以下の動画を選択してください。")
        
    else:
        # --- 動画の読み込み準備 ---
        # Streamlitが受け取った動画データを、Pythonが読み込めるように一時ファイルとして保存します
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        # OpenCVで動画を開く
        cap = cv2.VideoCapture(tfile.name)
        
        # 【追加】② 動画の長さ（秒数）のチェック
        # OpenCVのプロパティから「総フレーム数」と「FPS（1秒あたりのフレーム数）」を取得して計算
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 初期値の安全策（万が一FPSが0などで取得できなかった場合のクラッシュ防止）
        duration_sec = 0
        if fps > 0:
            duration_sec = total_frames / fps
            
        if duration_sec > MAX_VIDEO_DURATION_SEC:
            st.error(f"❌ 動画の長さが長すぎます（現在の動画: {duration_sec:.1f}秒）。サーバー負荷軽減のため、{MAX_VIDEO_DURATION_SEC}秒以内の動画を選択してください。")
            cap.release()  # キャプチャを解放
            st.rerun()
            
        else:
            # --- 💡 すべてのチェックをクリアした場合のみ、ここから解析処理が始まります ---
            # 画面にパラパラ漫画を映し出すための「空の枠」を用意
            frame_placeholder = st.empty()
            # 処理中のメッセージ
            st.info("🔄 動画を解析して再生中...（途中で止める場合はブラウザの「Stop」ボタンを押してください）")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or pd.isna(fps):
            fps = 30.0  # 万が一取得できなかった場合のデフォルト値

        frame_count = 0  # 現在のフレーム数を数えるカウンター
        tracking_logs = []  # 📊 [追加] 追跡ログを保存する空のリスト
        # ----------------------------------------------------------------------
        # --- [追加] セッション状態（記憶）の初期化 ---
        # アプリ起動時に、解析完了フラグとログの保存先を準備します
        if "analysis_done" not in st.session_state:
            st.session_state.analysis_done = False
        if "tracking_logs" not in st.session_state:
            st.session_state.tracking_logs = []
        # --------------------------------------------
        st.subheader("📊 現在のリアルタイム検知状況")
        col1, col2 = st.columns(2)
        car_metric = col1.metric(label="🚗 自動車 (画面内)", value=0)
        bike_metric = col2.metric(label="🚲 自転車 (画面内)", value=0)
        
        if not st.session_state.analysis_done:
            # 【追加】クラウド対策の設定（サイドバー等から変更できるようにしてもOK）
            FRAME_SKIP = frame_skip       # 何フレームに1回処理するか（3なら3フレームに1回 = 10fps化）
            TARGET_WIDTH = target_width   # リサイズ後の横幅（アスペクト比を維持して縮小）

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or pd.isna(fps):
                fps = 30.0

            frame_count = 0
            temp_logs = []  # ループ中は一時的なリストに溜める
            # ---------------------------------------------------------
            # 動画の全フレームをループ処理
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # 動画が終了したらループを抜ける

                frame_count += 1  # フレーム数をカウントアップ

                 # 💡 【追加】フレーム間引き処理
                # 設定した FRAME_SKIP の倍数のフレーム以外は、YOLOの推論をスキップして次のループへ
                if frame_count % FRAME_SKIP != 0:
                    continue  # 推論や描画をせず、次のフレーム読み込みへスキップ
                current_time_sec = round(frame_count / fps, 2)  # 経過時間（秒）を計算

                # 💡 【追加】動画のリサイズ処理
                # 元の画質が大きい場合、YOLOに渡す前、および画面表示用にサイズを小さくする
                h, w = frame.shape[:2]
                if w > TARGET_WIDTH:
                    # アスペクト比を維持して縮小後の高さを計算
                    target_height = int(h * (TARGET_WIDTH / w))
                    frame = cv2.resize(frame, (TARGET_WIDTH, target_height))
                # model.trackの処理
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
                    
                    # --- [追加] 各物体の固有ID(Track ID)を取得する ---
                    # 稀にIDがまだ付与されていないフレームがあるため、安全に取得します
                    track_ids = []
                    if result.boxes.id is not None:
                        track_ids = result.boxes.id.int().cpu().tolist()
                    else:
                        track_ids = [-1] * len(class_ids) # IDがない場合は-1にする
                    # -----------------------------------------------

                    for cid, tid in zip(class_ids, track_ids):
                        label = ""
                        if cid in [2, 5, 7]:
                            current_car_count += 1
                            label = "car"
                        elif cid == 1:
                            current_bike_count += 1
                            label = "bicycle"
                        
                        # --- 📊 [追加] 検知された物体のログを1件ずつリストに記録 ---
                        if label != "":
                            # 修正前: tracking_logs.append({
                                temp_logs.append({
                                "Frame": frame_count,
                                "Time(sec)": current_time_sec,
                                "Track_ID": tid,
                                "Class": label
                            })
                        # -------------------------------------------------------

                car_metric.metric(label="🚗 自動車 (画面内)", value=current_car_count)
                bike_metric.metric(label="🚲 自転車 (画面内)", value=current_bike_count)

                # 画面への表示（サイズが小さくなっているので表示も軽くなります）
                annotated_frame = result.plot()
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_frame_rgb, use_container_width=True)

            # 使い終わった動画ファイルを閉じて、一時ファイルを削除
            cap.release()
            os.unlink(tfile.name)
            # --- [追加] 解析が終わったら結果を「記憶」に保存し、フラグをONにする ---
            st.session_state.tracking_logs = temp_logs
            st.session_state.analysis_done = True
            # 画面をリフレッシュして、下のCSVダウンロード部品をすぐに表示させる
            st.rerun()

        # ======================================================================
        # --- 📊 [追加] ループ終了後（動画解析完了後）のCSVダウンロード処理 ---
        # ======================================================================
        if st.session_state.analysis_done and st.session_state.tracking_logs:
            st.success("🎉 動画の解析が完了しました！")
            
            # 1. ログのリストをデータフレームに変換
            df_logs = pd.DataFrame(st.session_state.tracking_logs)
            
            # データフレームが空ではない（列が存在する）かチェック
            if not df_logs.empty and "Class" in df_logs.columns:
                # 固有の（ユニークな）車の台数をカウント
                total_unique_cars = df_logs[df_logs["Class"] == "car"]["Track_ID"].nunique()
                total_unique_bikes = df_logs[df_logs["Class"] == "bicycle"]["Track_ID"].nunique()
                
                # 画面に計測結果を表示
                st.write(f"🚗 検知された自動車の総数（ユニーク）: {total_unique_cars} 台")
                st.write(f"🚲 検知された自転車の総数（ユニーク）: {total_unique_bikes} 台")
                
                # 【新機能】時間経過ごとの検知台数の推移グラフを表示（秒数丸め＋整数化版）
                st.subheader("⏱️ 時間経過ごとの検知台数の推移")
                
                # 1. 「Time(sec)」列が存在することを確認
                if "Time(sec)" in df_logs.columns and "Track_ID" in df_logs.columns:
                    time_unit = 1.0 
                    df_chart_prep = df_logs.copy()
                    df_filtered = df_chart_prep[df_chart_prep["Class"].isin(["car", "bicycle"])]
                    
                    if not df_filtered.empty:
                        df_frame_counts = df_filtered.groupby(["Time(sec)", "Class"])["Track_ID"].nunique().reset_index()
                        df_frame_counts["Time_Rounded"] = (df_frame_counts["Time(sec)"] / time_unit).round() * time_unit
                        df_rounded_counts = df_frame_counts.groupby(["Time_Rounded", "Class"])["Track_ID"].max().astype(int)
                        
                        # --- 💡 ここから Altair 用に修正・追加 ---
                        # 処理をしやすくするため、unstackせずに「縦持ちデータ（Long-form）」のままリセットインデックスします
                        df_chart_data = df_rounded_counts.reset_index()
                        df_chart_data.columns = ["Time(sec)", "Class", "Count"] # 列名を分かりやすく整える
                                                
                        # 1. クラスごとのカラーマッピングを厳密に定義
                        # データ内の文字列（"car", "bicycle"）と色（青, 赤）を完全に紐付けます
                        color_scale = alt.Scale(
                            domain=["car", "bicycle"],
                            range=["#0000FF", "#FF0000"]
                        )
                        
                        # 2. Altairで折れ線グラフを構築
                        chart = alt.Chart(df_chart_data).mark_line(strokeWidth=3).encode(
                            # X軸の設定（タイトルやグリッドを細かく制御可能）
                            x=alt.X("Time(sec):Q", title="経過時間 (秒)", axis=alt.Axis(grid=True)),
                            
                            # Y軸の設定（検知台数、整数のみ表示）
                            y=alt.Y("Count:Q", title="検知台数 (台)", axis=alt.Axis(format="d")),
                            
                            # 線の色（上記で定義したルールを適用）
                            color=alt.Color("Class:N", title="検知クラス", scale=color_scale),
                            
                            # 🔥 ポートフォリオで映える「ホバー（Tooltip）」機能の追加！
                            # マウスを乗せたときに、秒数・クラス・台数をポップアップ表示します
                            tooltip=[
                                alt.Tooltip("Time(sec):Q", title="時間(秒)"),
                                alt.Tooltip("Class:N", title="対象"),
                                alt.Tooltip("Count:Q", title="検知台数(台)")
                            ]
                        ).properties(
                            width="container", # 横幅を画面にフィットさせる
                            height=350         # グラフの高さを指定
                        ).interactive()        # グラフのマウスドラッグでの拡大・縮小・移動を有効化
                        
                        # 3. Streamlitの画面にAltairチャートを表示
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.info("グラフを表示するための自動車・自転車のデータが不足しています。")
                else:
                    st.error("ログデータに 'Time(sec)' または 'Track_ID' 列が存在しないため、グラフを描画できません。")
                # =================================================================                # =================================================================
                
                # --- レイアウト用のカラムを作成（ボタンを横並び、または縦に綺麗に並べるため） ---
                col1, col2 = st.columns(2)
                
                with col1:
                    # 既存のCSVダウンロードボタン
                    csv = df_logs.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="CSVをダウンロード",
                        data=csv,
                        file_name="tracking_results.csv",
                        mime="text/csv",
                        use_container_width=True # ボタンの横幅を揃えて綺麗に見せます
                    )
                    
                with col2:
                    #  【追加】同じ動画で再解析するためのボタン
                    if st.button("🔄 同じ動画で再解析する", use_container_width=True):
                        # フラグをFalseに戻すことで、上の while ループがもう一度走るようになります
                        st.session_state.analysis_done = False
                        # ログを一度空っぽにしてリセット
                        st.session_state.tracking_logs = []
                        # 画面を即座にリフレッシュして再解析の実行へ移る
                        st.rerun()
        else:
            st.warning("⚠️ 動画内で物体は検知されなかったため、ダウンロードできるCSVデータはありません。")
            st.write("🚗 検知された自動車の総数: 0 台")
            st.write("🚲 検知された自転車の総数: 0 台")
            
            #  【追加】検知されなかった場合でも、設定を変えてやり直せるようにボタンを配置
            if st.button("🔄 設定を変えて再解析する", use_container_width=True):
                st.session_state.analysis_done = False
                st.session_state.tracking_logs = []
                st.rerun()
            # ======================================================================