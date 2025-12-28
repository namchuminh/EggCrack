import sys
sys.path.append('ultralytics')
import os
import time
import numpy as np
import cv2
import tensorflow as tf
from ultralytics import YOLO
from PyQt5 import QtCore, QtGui, QtWidgets

# ================= CONFIG =================
YOLO_MODEL = "best.onnx"
CLASSIFIER_MODEL = "mobilenetv2_egg_classifier.keras"

IMGSZ = 640
CONF_THRESH = 0.1
CLASSIFIER_IMG_SIZE = (256, 256)
MIN_ROI_SIZE = 0

BROKEN_THRESH = 0.99
ROI_PAD_RATIO = 0.0

# Stability rule:
# Need N consecutive frames with same prediction to lock label for that track_id
STABLE_FRAMES = 3

# If False: once locked, never change again
# If True : allow changing, but also requires N consecutive opposite frames
ALLOW_RELOCK = False
# =========================================


# ================= VIDEO WORKER =================
class VideoWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage)
    status_ready = QtCore.pyqtSignal(str)

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.running = True
        self.paused = False

        # lock state
        self.id_lock = {}  # track_id -> locked label 0/1
        self.id_run = {}   # track_id -> {'last':0/1, 'cnt':int}

    def stop(self):
        self.running = False

    def pause(self, p):
        self.paused = p

    def run(self):
        try:
            yolo = YOLO(YOLO_MODEL)
        except Exception as e:
            self.status_ready.emit(f"❌ Lỗi YOLO: {e}")
            return

        try:
            clf = tf.keras.models.load_model(CLASSIFIER_MODEL)
        except Exception as e:
            self.status_ready.emit(f"❌ Lỗi Classifier: {e}")
            return

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.status_ready.emit("❌ Không mở được nguồn video/camera.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1:
            fps = 30.0

        # reset
        self.id_lock.clear()
        self.id_run.clear()
        self.status_ready.emit("🟢 ĐANG CHẠY")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            out = frame.copy()
            h0, w0 = frame.shape[:2]
            if w0 != w or h0 != h:
                w, h = w0, h0

            # detect + track on full frame (no zone)
            results = yolo.track(frame, imgsz=IMGSZ, conf=CONF_THRESH, persist=True, verbose=False)

            if results and results[0].boxes:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id[0]) if box.id is not None else None

                    if (x2 - x1) < MIN_ROI_SIZE or (y2 - y1) < MIN_ROI_SIZE:
                        continue

                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w - 1, x2); y2 = min(h - 1, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # ROI pad
                    pad = int(ROI_PAD_RATIO * max(x2 - x1, y2 - y1))
                    xx1 = max(0, x1 - pad); yy1 = max(0, y1 - pad)
                    xx2 = min(w - 1, x2 + pad); yy2 = min(h - 1, y2 + pad)

                    roi = frame[yy1:yy2, xx1:xx2]
                    if roi.size == 0:
                        continue

                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    roi = cv2.resize(roi, CLASSIFIER_IMG_SIZE, interpolation=cv2.INTER_AREA)

                    # match training: 0..255, no /255
                    x = roi.astype(np.float32)
                    x = np.expand_dims(x, axis=0)

                    prob = float(clf.predict(x, verbose=0)[0][0])
                    cur_state = 1 if prob > BROKEN_THRESH else 0  # 1=VỠ, 0=NGUYÊN

                    # stable lock
                    show_state = cur_state
                    if track_id is not None:
                        if track_id in self.id_lock:
                            locked = self.id_lock[track_id]
                            show_state = locked

                            if ALLOW_RELOCK and cur_state != locked:
                                run = self.id_run.get(track_id)
                                if run is None:
                                    run = {"last": cur_state, "cnt": 1}
                                else:
                                    if run["last"] == cur_state:
                                        run["cnt"] += 1
                                    else:
                                        run["last"] = cur_state
                                        run["cnt"] = 1
                                self.id_run[track_id] = run

                                if run["cnt"] >= STABLE_FRAMES:
                                    self.id_lock[track_id] = cur_state
                                    self.id_run[track_id] = {"last": cur_state, "cnt": 0}
                                    show_state = cur_state
                        else:
                            run = self.id_run.get(track_id)
                            if run is None:
                                run = {"last": cur_state, "cnt": 1}
                            else:
                                if run["last"] == cur_state:
                                    run["cnt"] += 1
                                else:
                                    run["last"] = cur_state
                                    run["cnt"] = 1
                            self.id_run[track_id] = run

                            if run["cnt"] >= STABLE_FRAMES:
                                self.id_lock[track_id] = cur_state
                                self.id_run[track_id] = {"last": cur_state, "cnt": 0}
                                show_state = cur_state

                    label = "TRUNG_VO" if show_state == 1 else "TRUNG_NGUYEN"
                    color = (70, 140, 255) if show_state == 1 else (40, 220, 120)

                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                    tag = f"{label}  {prob:.2f}"
                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    y_text = max(28, y1 - 10)
                    cv2.rectangle(out, (x1, y_text - th - 10), (x1 + tw + 14, y_text + 6), color, -1)
                    cv2.putText(out, tag, (x1 + 7, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (15, 15, 15), 2)

            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                                QtGui.QImage.Format_RGB888).copy()
            self.frame_ready.emit(qimg)

            time.sleep(max(0.0, 1.0 / fps))

        cap.release()
        self.status_ready.emit("⛔ ĐÃ DỪNG")


# ================= MAIN WINDOW =================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EGG INSPECTION SYSTEM")
        self.resize(1200, 780)

        self.video_path = None
        self.worker = None

        self._build_ui()
        self._wire()

        self._set_status("CHƯA CHỌN VIDEO / CAMERA")
        self._set_running(False)

    def _build_ui(self):
        self.setStyleSheet("""
            QMainWindow { background: #0B0F17; }
            QLabel { color: #E6EDF7; font-family: Inter, Arial; }
            QPushButton {
                background: #1C2536;
                color: #E6EDF7;
                border: 1px solid #2B3B57;
                padding: 10px 12px;
                border-radius: 10px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover { background: #22304A; }
            QPushButton:disabled { color:#6C7A96; border-color:#1A2436; background:#121A28; }
            QPushButton#primary { background:#2E6BFF; border-color:#2E6BFF; }
            QPushButton#primary:hover { background:#255AE0; }
            QPushButton#danger { background:#E84C4C; border-color:#E84C4C; }
            QPushButton#danger:hover { background:#D44343; }
            QPushButton#warn { background:#F2B84B; border-color:#F2B84B; color:#0B0F17; }
            QPushButton#warn:hover { background:#E2A93F; }
            QPushButton#success { background:#31C48D; border-color:#31C48D; color:#0B0F17; }
            QPushButton#success:hover { background:#2AB07E; }
            QFrame#card {
                background: #0F1624;
                border: 1px solid #1D2A41;
                border-radius: 14px;
            }
            QLabel#h1 { font-size: 18px; font-weight: 800; }
            QLabel#muted { color: #9FB0CC; }
            QLabel#pill {
                background:#121C2D;
                border:1px solid #1D2A41;
                padding:10px 12px;
                border-radius:12px;
                font-size: 14px;
                font-weight: 700;
            }
        """)

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(12)

        header = QtWidgets.QFrame()
        header.setObjectName("card")
        header.setProperty("id", "card")
        hL = QtWidgets.QHBoxLayout(header)
        hL.setContentsMargins(16, 14, 16, 14)

        lblTitle = QtWidgets.QLabel("🥚 EGG INSPECTION")
        lblTitle.setObjectName("h1")
        lblTitle.setProperty("id", "h1")

        lblSub = QtWidgets.QLabel("YOLO detect + MobileNetV2 classify | No stats | No zone")
        lblSub.setObjectName("muted")
        lblSub.setProperty("id", "muted")

        tbox = QtWidgets.QVBoxLayout()
        tbox.setSpacing(2)
        tbox.addWidget(lblTitle)
        tbox.addWidget(lblSub)

        self.lblStatus = QtWidgets.QLabel("...")
        self.lblStatus.setObjectName("pill")
        self.lblStatus.setProperty("id", "pill")
        self.lblStatus.setAlignment(QtCore.Qt.AlignCenter)

        hL.addLayout(tbox, 1)
        hL.addWidget(self.lblStatus, 0)
        outer.addWidget(header)

        body = QtWidgets.QFrame()
        body.setObjectName("card")
        body.setProperty("id", "card")
        bL = QtWidgets.QVBoxLayout(body)
        bL.setContentsMargins(16, 16, 16, 16)
        bL.setSpacing(12)

        self.lblCamera = QtWidgets.QLabel("NO VIDEO / CAMERA")
        self.lblCamera.setAlignment(QtCore.Qt.AlignCenter)
        self.lblCamera.setMinimumSize(920, 620)
        self.lblCamera.setStyleSheet("""
            QLabel {
                background: #05070D;
                border: 1px solid #1D2A41;
                border-radius: 14px;
            }
        """)

        btnRow = QtWidgets.QHBoxLayout()
        btnRow.setSpacing(10)

        self.btnSelect = QtWidgets.QPushButton("📂 Chọn video")
        self.btnCam0 = QtWidgets.QPushButton("📷 Camera 0")
        self.btnCam0.setObjectName("success")
        self.btnCam0.setProperty("id", "success")

        self.btnStart = QtWidgets.QPushButton("▶ Start video")
        self.btnStart.setObjectName("primary")
        self.btnStart.setProperty("id", "primary")

        self.btnPause = QtWidgets.QPushButton("⏸ Pause")
        self.btnPause.setObjectName("warn")
        self.btnPause.setProperty("id", "warn")

        self.btnStop = QtWidgets.QPushButton("⏹ Stop")
        self.btnStop.setObjectName("danger")
        self.btnStop.setProperty("id", "danger")

        btnRow.addWidget(self.btnSelect)
        btnRow.addWidget(self.btnCam0)
        btnRow.addStretch(1)
        btnRow.addWidget(self.btnStart)
        btnRow.addWidget(self.btnPause)
        btnRow.addWidget(self.btnStop)

        bL.addWidget(self.lblCamera, 1)
        bL.addLayout(btnRow)

        outer.addWidget(body, 1)

    def _wire(self):
        self.btnSelect.clicked.connect(self.select_video)
        self.btnCam0.clicked.connect(self.open_camera0)
        self.btnStart.clicked.connect(self.start_video)
        self.btnPause.clicked.connect(self.pause)
        self.btnStop.clicked.connect(self.stop)

    def _set_status(self, s):
        self.lblStatus.setText(s)

    def _set_running(self, running: bool):
        self.btnStop.setEnabled(running)
        self.btnPause.setEnabled(running)
        self.btnStart.setEnabled(not running)
        self.btnCam0.setEnabled(not running)
        self.btnSelect.setEnabled(not running)

    def _start_worker(self, source, status_text):
        self.stop()
        self.worker = VideoWorker(source)
        self.worker.frame_ready.connect(self.show_frame)
        self.worker.status_ready.connect(self._set_status)
        self.worker.start()
        self._set_status(status_text)
        self._set_running(True)

    def select_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Chọn video", "", "Video Files (*.mp4 *.avi *.mkv)"
        )
        if path:
            self.video_path = path
            self._set_status(f"ĐÃ CHỌN: {os.path.basename(path)}")

    def open_camera0(self):
        self.video_path = None
        self._start_worker(0, "🟢 CAMERA 0 ĐANG CHẠY")

    def start_video(self):
        if not self.video_path:
            QtWidgets.QMessageBox.warning(self, "Lỗi", "Chưa chọn video!")
            return
        self._start_worker(self.video_path, "🟢 VIDEO ĐANG CHẠY")

    def pause(self):
        if not self.worker:
            return
        self.worker.pause(not self.worker.paused)
        self._set_status("⏸ PAUSED" if self.worker.paused else "🟢 ĐANG CHẠY")

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(1500)
            self.worker = None
        self._set_running(False)
        if self.video_path:
            self._set_status(f"ĐÃ CHỌN: {os.path.basename(self.video_path)}")
        else:
            self._set_status("⛔ ĐÃ DỪNG")

    def show_frame(self, img):
        pix = QtGui.QPixmap.fromImage(img)
        self.lblCamera.setPixmap(
            pix.scaled(self.lblCamera.size(),
                       QtCore.Qt.KeepAspectRatio,
                       QtCore.Qt.SmoothTransformation)
        )


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
