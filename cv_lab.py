#!/usr/bin/env python3
"""
cv_lab.py - OpenCV 선 따라가기 짝 프로그래밍 실습

할 일: [C1] ~ [C5] 함수의 TODO 작성 (둘이 함께, 분담은 자유)
함수 하나를 끝낼 때마다  python3 check_cv.py  로 확인

카메라 사진에서 검은 선을 찾아 조향을 계산하고, 실제로 차를 움직인다.
텀 프로젝트의 OpenCV 자율주행과 자동 데이터 수집의 기초가 되는 함수들.

  [C1] make_mask()     위쪽 자르기 → 흑백 → 흐리게 → 이진화 (검은 선 → 흰색)
  [C2] find_line_x()   띠(가로 줄) 하나에서 선의 x 위치
  [C3] line_error()    선 위치 → 오차 -1 ~ 1
  [C4] pd_steer()      오차 → 조향 (PD 제어)
  [C5] save_sample()   사진 + labels.csv 한 줄 (학습 데이터)
  [추가 1] 모터 온도 감속   motor_serial.ino의 checkMotorTemp() (여기서는 상태를 받아 웹 화면에 표시)
  [추가 2] find_line_points() 3개 지점에서 선 찾기,  crosswalk_stop() 횡단보도에서 멈췄다 가기
  [추가 3] pid_steer()         PID 제어 (PD + I + D 잡음 거르기)
  추가 기능은 아래 FEATURES 목록에서 켜고 끔
  [제공] 카메라, 웹 화면, 모터(motor_lab의 mix, to_command, MotorLink), 주행 반복

준비 (Pi 5, 한 번만) - 자세한 내용은 README.txt:
  sudo apt install -y python3-venv python3-picamera2 python3-opencv python3-numpy python3-flask python3-serial
  python3 -m venv --system-site-packages venv      # cv_lab 폴더에서
  아두이노에는 arduino/motor_serial/motor_serial.ino 업로드 (MIN_PWM · RIGHT_BOOST는 자기 차에 맞게)

실행 (Pi 5, 카메라 + 아두이노 USB + 모터 배터리):
  source venv/bin/activate
  python3 check_cv.py     # 먼저 카메라·모터 없이 C1 ~ C5 확인
  python3 cv_lab.py       # http://<Pi IP>:8000  →  '출발'
"""

import csv
import glob
import json
import os
import threading
import time

import cv2
import numpy as np

# ================= 기능 켜기 / 끄기 =================
# 끄려면 그 줄 앞에 # 를 붙여 주석 처리 (아두이노 쪽 기능은 motor_serial.ino 맨 위에서)
FEATURES = [
    "three_points",  # [추가 2] 3개 지점에서 선 찾기 (끄면 가운데 1지점만)
    # "crosswalk",  # [추가 2] 횡단보도에서 CROSSWALK_STOP_SEC 동안 정지
    "pid",  # [추가 3] PID 제어 (끄면 기존 PD: pd_steer)
    # "arduino_status",  # [추가 1] 아두이노가 보낸 온도 · 속도 · 배터리를 웹 화면에 표시
]
ALL_FEATURES = ("three_points", "crosswalk", "pid", "arduino_status")


def check_features(names):
    """오타로 기능이 조용히 꺼지지 않게, 모르는 이름이 있으면 바로 알림"""
    unknown = [n for n in names if n not in ALL_FEATURES]
    if unknown:
        raise ValueError(
            f"FEATURES에 모르는 이름 {unknown}. 쓸 수 있는 이름: {ALL_FEATURES}"
        )


check_features(FEATURES)


def on(name):
    return name in FEATURES


# ================= 설정 =================
SIZE = (320, 240)  # 카메라 해상도 (가로, 세로)
ROTATE_180 = True  # 카메라를 거꾸로 달았으면 True (화면이 뒤집혀 보이면 반대로)
PORT = 8000
SETTINGS = {  # 웹 화면 슬라이더로 바꾸는 값 (처음 값)
    "thresh": 80,  # 밝기 문턱: 이보다 어두우면 선
    "roi_top": 0.1,  # 위쪽에서 버리는 비율
    "look": 0.5,  # 살펴볼 띠 위치 (자른 영역 안, 0 = 위, 1 = 아래)
    "kp": 1.0,  # P: 벗어난 만큼 꺾기
    "ki": 0.1,  # I: 한쪽으로 계속 치우치는 것 보정 (0 = 끄기)
    "kd": 0.5,  # D: 좌우로 흔들리는 것 억제
    "speed": 0.35,
}
SLOWDOWN = 0.4  # 꺾을 때 속도 줄이기: 실제 속도 = 속도 × (1 − SLOWDOWN × |조향|)
LOST_STOP = (
    6  # 선을 놓친 동안 이 프레임 수 전까지는 하던 조향 그대로 (HOLD, 녹화 안 함),
)
# 이만큼 연속으로 못 찾으면 정지 (20fps에서 약 0.3초)
RECORD_FPS = 10

# --- PID ---
I_LIMIT = 0.3  # I가 조향에 더할 수 있는 최대값 (너무 쌓이지 않게)
D_FILTER_SEC = (
    0.05  # D를 부드럽게 하는 시간 (초). 0 = 거르지 않음, 크면 부드럽지만 반응이 늦음
)

# --- 3개 지점 ---
LOOK_OFFSETS = (-0.30, 0.0, 0.30)  # 슬라이더 look 기준 위치 (먼 곳, 가운데, 가까운 곳)
LOOK_WEIGHTS = (1.0, 2.0, 1.0)  # 조향을 정할 때 각 지점의 비중

# --- 횡단보도 ---
CROSSWALK_STOP_SEC = 3.0  # 횡단보도 앞에서 멈추는 시간 (초)
CROSSWALK_MIN_LINES = 3  # 한 지점에서 선이 이 개수 이상이면 횡단보도 무늬
CROSSWALK_MIN_POINTS = 1  # 3개 지점 중 이 개수 이상에서 무늬가 보여야 횡단보도
CROSSWALK_FRAMES = 3  # 이만큼 연속으로 보여야 정지 (잘못 본 한두 장은 무시)
CROSSWALK_CLEAR_SEC = (
    1.0  # 다시 출발한 뒤 무늬가 이 시간 동안 안 보여야 다음 횡단보도를 찾음
)
MIN_LINE_WIDTH = 3  # 이보다 좁은 것은 선으로 세지 않음 (픽셀)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", time.strftime("%Y%m%d_%H%M%S") + "_lab")


# =====================================================================
# [C1] 선만 흰색으로 남긴 흑백 이미지
#   bgr: 카메라 사진 (세로, 가로, 3)
#   1) 위쪽 roi_top 비율을 잘라냄 (y0 = 자른 줄)
#   2) 흑백 → 흐리게(5 × 5) → 이진화: thresh보다 어두우면 255, 밝으면 0
#   3) 작은 점 지우기 (3 × 3 열림 연산)
#   반환: (mask, y0)   mask 크기 = (세로 − y0, 가로), 값은 0 또는 255
# =====================================================================
def make_mask(bgr, roi_top, thresh):
    height = bgr.shape[0]
    y0 = int(height * roi_top)

    # 관심 영역을 자르고 흑백으로 변환한 뒤, 약한 노이즈를 흐리게 한다.
    roi = bgr[y0:]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 어두운 픽셀(검은 선)을 흰색으로 반전 이진화한다.
    _, mask = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY_INV)

    # 3x3 열림 연산으로 작은 흰색 잡티를 제거한다.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask, y0


# =====================================================================
# [C2] 띠 하나에서 선의 x 위치
#   띠: mask에서 세로 위치 look(0 ~ 1)을 가운데로, 높이의 band 비율만큼
#   띠 안에서 열(세로 줄)마다 흰 점 개수를 세고, 띠 높이의 1/4 이상인 열을 '선'으로 봄
#   반환: 선인 열들의 x 평균 (float),  선이 없으면 None
#   예: 띠 안 x = 194 ~ 206 이 흰색 → 200.0
# =====================================================================
def find_line_x(mask, look, band=0.16):
    height = mask.shape[0]
    b0 = max(0, int(height * (look - band / 2)))
    b1 = min(height, int(height * (look + band / 2)))

    if b1 <= b0:
        return None

    strip = mask[b0:b1]
    white_counts = (strip > 0).sum(axis=0)
    line_columns = np.where(white_counts >= (b1 - b0) / 4)[0]

    if len(line_columns) == 0:
        return None
    return float(line_columns.mean())


# =====================================================================
# [C3] 선 위치 → 오차
#   오차 = (선 x − 화면 중심) ÷ (화면 폭의 절반)
#   예: 폭 320에서 x 160 → 0,  x 240 → +0.5 (선이 오른쪽),  x 0 → −1
# =====================================================================
def line_error(x, width):
    return (x - width / 2) / (width / 2)


# =====================================================================
# [C4] PD 제어: 오차 → 조향 (−1 ~ 1)
#   변화 = (오차 − 이전 오차) ÷ dt × 0.1        (0.1초 동안의 변화, dt가 0 이하면 0)
#   조향 = kp × 오차 + kd × 변화,  −1 ~ 1로 자르기
#   예: pd_steer(0.5, 0.5, 0.05, 1.0, 0.5) → 0.5
#       pd_steer(0.5, 0.4, 0.1, 1.0, 0.5)  → 0.5 + 0.5 × 0.1 = 0.55
# =====================================================================
def pd_steer(error, prev_error, dt, kp, kd):
    change = (error - prev_error) / dt * 0.1 if dt > 0 else 0.0
    steer = kp * error + kd * change
    return max(-1.0, min(1.0, steer))


# =====================================================================
# [C5] 학습 데이터 한 장 저장
#   folder/images/000001.jpg 로 사진 저장 (index 6자리, JPEG 품질 90)
#   folder/labels.csv 에 한 줄 추가: image, steering, throttle, time
#   (파일이 없으면 먼저 머리줄 image,steering,throttle,time 을 씀)
#   반환: "images/000001.jpg"
# =====================================================================
def save_sample(folder, index, frame, steering, throttle):
    os.makedirs(os.path.join(folder, "images"), exist_ok=True)
    image_path = f"images/{index:06d}.jpg"
    full_image_path = os.path.join(folder, image_path)

    if not cv2.imwrite(full_image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise IOError(f"사진 저장 실패: {full_image_path}")

    labels_path = os.path.join(folder, "labels.csv")
    write_header = not os.path.exists(labels_path)
    with open(labels_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(["image", "steering", "throttle", "time"])
        writer.writerow([image_path, steering, throttle, time.time()])

    return image_path


# =====================================================================
# [추가 2] 3개 지점 인식 + 횡단보도
# =====================================================================
def count_lines(mask, look, band=0.16):
    """띠 하나에서 서로 떨어진 선의 개수 (선으로 보는 기준은 find_line_x와 같음)"""
    height = mask.shape[0]
    b0 = max(0, int(height * (look - band / 2)))
    b1 = min(height, int(height * (look + band / 2)))
    if b1 <= b0:
        return 0

    is_line = (mask[b0:b1] > 0).sum(axis=0) >= (b1 - b0) / 4
    # 선이 시작하는 열과 끝나는 열을 찾아 폭을 잰다.
    edges = np.diff(np.concatenate(([0], is_line.astype(np.int8), [0])))
    widths = np.where(edges == -1)[0] - np.where(edges == 1)[0]
    return int((widths >= MIN_LINE_WIDTH).sum())


def active_points():
    """켜진 기능에 따라 (지점 위치들, 비중들). three_points를 끄면 슬라이더 look 한 곳만"""
    if on("three_points"):
        return LOOK_OFFSETS, LOOK_WEIGHTS
    return (0.0,), (1.0,)


def find_line_points(mask, look):
    """3개 지점에서 선 찾기. 반환: [(look, x, 선 개수), ...] (먼 곳부터)
    선이 없거나 횡단보도 무늬인 지점은 x = None (조향에 쓰지 않음)"""
    points = []
    for offset in active_points()[0]:
        pos = max(0.08, min(0.92, look + offset))  # 띠가 화면 밖으로 나가지 않게
        lines = count_lines(mask, pos)
        pattern = on("crosswalk") and lines >= CROSSWALK_MIN_LINES
        x = None if pattern else find_line_x(mask, pos)
        points.append((pos, x, lines))
    return points


def combine_points(points):
    """찾은 지점들의 x를 비중(LOOK_WEIGHTS)으로 가중 평균. 하나도 없으면 None"""
    weights = active_points()[1]
    found = [(x, w) for (_, x, _), w in zip(points, weights) if x is not None]
    if not found:
        return None
    return sum(x * w for x, w in found) / sum(w for _, w in found)


def is_crosswalk(points):
    """횡단보도 무늬(선 여러 개)가 보이는 지점이 충분히 많으면 True (crosswalk를 끄면 항상 False)"""
    if not on("crosswalk"):
        return False
    seen = sum(1 for _, _, lines in points if lines >= CROSSWALK_MIN_LINES)
    return seen >= CROSSWALK_MIN_POINTS


def new_crosswalk_state():
    return {"mode": "drive", "count": 0, "until": 0.0, "last_seen": 0.0}


def crosswalk_stop(cw, seen, now):
    """횡단보도를 만나면 CROSSWALK_STOP_SEC 동안 True(정지)를 반환.
    cw["mode"]:  drive (주행) → stop (정지) → pass (지나가는 중, 같은 횡단보도에 다시 안 멈춤) → drive"""
    if seen:
        cw["last_seen"] = now

    if cw["mode"] == "drive":
        cw["count"] = cw["count"] + 1 if seen else 0
        if cw["count"] >= CROSSWALK_FRAMES:
            cw["mode"], cw["until"] = "stop", now + CROSSWALK_STOP_SEC
    elif cw["mode"] == "stop":
        if now >= cw["until"]:
            cw["mode"], cw["last_seen"] = "pass", now
    elif now - cw["last_seen"] >= CROSSWALK_CLEAR_SEC:
        cw["mode"], cw["count"] = "drive", 0

    return cw["mode"] == "stop"


# =====================================================================
# [추가 3] PID 제어: 오차 → 조향 (−1 ~ 1)
#   P = kp × 오차
#   I = ki × 오차를 시간에 따라 더한 값  (±I_LIMIT로 제한, accumulate가 True일 때만 쌓음)
#   D = kd × 변화  (pd_steer와 같은 0.1초 기준, D_FILTER_SEC로 잡음을 거름)
#   dt가 0 이하(처음이거나 선을 놓쳤다 다시 찾음)면 D = 0, I는 그대로
# =====================================================================
def new_pid_state():
    return {"prev": 0.0, "i": 0.0, "d": 0.0}


def pid_steer(pid, error, dt, kp, ki, kd, accumulate=True):
    if dt > 0:
        change = (error - pid["prev"]) / dt * 0.1
        pid["d"] += (change - pid["d"]) * dt / (D_FILTER_SEC + dt)
    else:
        pid["d"] = 0.0
    pid["prev"] = error

    if ki == 0:
        pid["i"] = 0.0
    pd = kp * error + kd * pid["d"]
    # 조향이 이미 끝까지 꺾였으면 같은 방향으로는 더 쌓지 않는다.
    saturated = abs(pd + pid["i"]) >= 1.0 and error * (pd + pid["i"]) > 0
    if accumulate and dt > 0 and not saturated:
        pid["i"] = max(-I_LIMIT, min(I_LIMIT, pid["i"] + ki * error * dt))

    return max(-1.0, min(1.0, pd + pid["i"]))


def compute_steer(pid, error, dt, s, moving):
    """pid 기능이 켜져 있으면 pid_steer, 꺼져 있으면 기존 pd_steer"""
    if on("pid"):
        # I는 차가 실제로 움직이던 동안만 쌓는다 (정지 중에 쌓이면 출발할 때 튐).
        return pid_steer(pid, error, dt, s["kp"], s["ki"], s["kd"], accumulate=moving)
    steer = pd_steer(error, pid["prev"], dt, s["kp"], s["kd"])
    pid["prev"] = error
    return steer


# ---------------------------------------------------------------------
# [제공] 모터 (motor_lab 실습에서 만든 함수와 같음)
# ---------------------------------------------------------------------
def mix(speed, steer):
    left, right = speed * (1 + steer), speed * (1 - steer)
    m = max(abs(left), abs(right), 1.0)
    return left / m, right / m


def to_command(left, right):
    l = int(round(max(-1.0, min(1.0, left)) * 255))
    r = int(round(max(-1.0, min(1.0, right)) * 255))
    return f"M,{l},{r}\n"


class MotorLink:
    def __init__(self, port=None, baud=115200):
        import serial  # sudo apt install python3-serial

        ports = sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*"))
        self.port = port or (ports[0] if ports else None)
        if self.port is None:
            raise SystemExit("아두이노를 찾을 수 없음: USB 연결 확인")
        self.ser = serial.Serial(self.port, baud, timeout=0.5)
        time.sleep(2)  # 포트를 열면 아두이노가 재시작됨
        self.ser.reset_input_buffer()
        self.rx = ""  # 아직 줄이 끝나지 않은 수신 글자
        self.status = ""  # 아두이노가 보낸 마지막 상태 (TEMP=...)

    def send(self, text):
        self.ser.write(text.encode())

    def read_status(self):
        """[추가 1] 도착한 글자만 읽고(기다리지 않음) 마지막 상태 줄을 반환"""
        n = self.ser.in_waiting
        if n:
            self.rx += self.ser.read(n).decode(errors="ignore")
            *lines, self.rx = self.rx.split("\n")
            for line in lines:
                if line.startswith("TEMP="):
                    self.status = line.strip()
        return self.status

    def close(self):
        try:
            self.send("M,0,0\n")
        finally:
            self.ser.close()


# ---------------------------------------------------------------------
# [제공] 화면에 그리기
# ---------------------------------------------------------------------
def draw(frame, y0, mask, points, steer, moving, note):
    view = frame.copy()
    h, w = view.shape[:2]
    cv2.line(view, (0, y0), (w, y0), (255, 255, 0), 1)  # 자르는 경계 (하늘색)
    if mask is not None:
        mh = mask.shape[0]
        for look, x, lines in points:
            b0 = y0 + int(mh * look - mh * 0.08)
            b1 = y0 + int(mh * look + mh * 0.08)
            # 살펴보는 띠 (초록, 횡단보도 무늬가 보이면 노랑)
            pattern = on("crosswalk") and lines >= CROSSWALK_MIN_LINES
            color = (0, 255, 255) if pattern else (0, 200, 0)
            cv2.rectangle(view, (0, b0), (w - 1, b1), color, 1)
            cv2.line(view, (w // 2, b0), (w // 2, b1), (255, 255, 255), 1)  # 화면 중심
            if x is not None:
                cv2.circle(
                    view, (int(x), (b0 + b1) // 2), 6, (0, 0, 255), -1
                )  # 찾은 선 (빨강)
    if steer is not None:
        tip = (int(w / 2 + steer * w * 0.4), int(h * 0.55))
        cv2.arrowedLine(
            view,
            (w // 2, h - 8),
            tip,
            (0, 255, 0) if moving else (0, 165, 255),
            3,
            tipLength=0.2,
        )
    cv2.putText(view, note, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    right = (
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if mask is not None
        else np.zeros((1, w, 3), np.uint8)
    )
    right = cv2.copyMakeBorder(
        right, h - right.shape[0], 0, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40)
    )
    return cv2.hconcat([view, right])


# ---------------------------------------------------------------------
# [제공] 카메라 + 주행 반복 + 웹 화면
# ---------------------------------------------------------------------
def full_view_size(cam):
    """센서 전체(가장 넓은 화각)를 쓰는 센서 모드 중 가장 작은 것의 크기. 예: Camera Module 2 → (1640, 1232)
    작은 화면(320 x 240)만 요청하면 카메라가 센서 가운데만 잘라 쓰는 모드를 골라 화면이 확대(좁게)되기 때문"""
    try:
        area = (
            lambda m: m["crop_limits"][2] * m["crop_limits"][3]
        )  # 센서에서 실제로 쓰는 넓이
        widest = max(area(m) for m in cam.sensor_modes)
        full = [m for m in cam.sensor_modes if area(m) >= widest * 0.99]
        return tuple(min(full, key=lambda m: m["size"][0] * m["size"][1])["size"])
    except Exception:
        return None  # 못 찾으면 카메라 기본 모드


def main():
    from flask import Flask, Response, jsonify, request
    from libcamera import Transform
    from picamera2 import Picamera2

    cam = Picamera2()
    raw = full_view_size(cam)
    print(f"카메라: 센서 모드 {raw or '기본'} → 화면 {SIZE}")
    cam.configure(
        cam.create_video_configuration(
            main={"size": SIZE, "format": "RGB888"},
            **({"raw": {"size": raw}} if raw else {}),  # 센서 전체 사용 (넓은 화각)
            transform=Transform(hflip=1, vflip=1) if ROTATE_180 else Transform(),
        )
    )
    cam.start()
    time.sleep(1)
    link = MotorLink()
    S = dict(SETTINGS)
    st = {
        "auto": False,
        "rec": False,
        "beat": 0.0,
        "view": None,
        "info": {},
        "frames": 0,
    }
    lock = threading.Lock()

    def loop():
        prev_t, lost, last_save = time.time(), 0, 0.0
        last_steer = None  # 마지막으로 선을 보고 계산한 조향
        had_err = False  # 바로 앞 프레임에서 선을 봤는지
        cw = new_crosswalk_state()
        pid, moving = new_pid_state(), False
        while True:
            frame = cam.capture_array("main")
            now = time.time()
            with lock:
                s = dict(S)
                if now - st["beat"] > 0.5:
                    st["auto"] = False  # 웹 화면 신호가 끊김 → 정지
                auto, rec = st["auto"], st["rec"]
            mask = x = err = steer = None
            points, crosswalk = [], False
            note = ""
            try:
                mask, y0 = make_mask(frame, s["roi_top"], s["thresh"])
                points = find_line_points(mask, s["look"])
                crosswalk = is_crosswalk(points)
                x = combine_points(points)
                if x is not None:
                    err = line_error(x, frame.shape[1])
                    dt = (
                        now - prev_t if had_err else 0.0
                    )  # 놓쳤다가 다시 찾은 첫 프레임: D 생략 (dt 0 → 변화 0)
                    steer = compute_steer(pid, err, dt, s, moving)
                elif crosswalk:
                    steer = 0.0  # 횡단보도 무늬만 보임 → 직진
            except NotImplementedError as e:
                note = f"{e} 아직"
                y0 = int(frame.shape[0] * s["roi_top"])
            prev_t = now
            had_err = err is not None
            lost = 0 if steer is not None else lost + 1
            holding = False
            if steer is not None:
                last_steer = steer
            elif last_steer is not None and lost < LOST_STOP:
                steer, holding = last_steer, True  # 잠깐 놓침: 하던 조향 그대로
            if not auto:
                cw = new_crosswalk_state()  # 정지 버튼을 누르면 처음부터
                pid["i"] = 0.0
            waiting = auto and crosswalk_stop(cw, crosswalk, now)
            moving = (
                auto and steer is not None and not waiting
            )  # 계속 놓치면 steer가 None → 정지
            if moving:
                left, right = mix(s["speed"] * (1 - SLOWDOWN * abs(steer)), steer)
            else:
                left = right = 0.0
            cmd = to_command(left, right)
            link.send(cmd)  # 매 프레임 전송 = 하트비트
            if (
                rec
                and moving
                and err is not None
                and now - last_save >= 1.0 / RECORD_FPS
            ):  # 선을 본 장면만 저장
                last_save = now
                try:
                    with lock:
                        st["frames"] += 1
                        n = st["frames"]
                    save_sample(DATA_DIR, n, frame, steer, s["speed"])
                except NotImplementedError:
                    note = "C5 아직"
            if not note:
                if waiting:
                    note = f"CROSSWALK STOP {cw['until'] - now:.1f}s"
                elif holding:
                    note = f"HOLD {lost} steer {steer:+.2f}"
                elif x is not None:
                    note = f"x {x:.0f} err {err:+.2f} steer {steer:+.2f}"
                else:
                    note = "CROSSWALK" if crosswalk else "LOST"
            view = draw(frame, y0, mask, points, steer, moving, note)
            status = link.read_status()  # 꺼져 있어도 읽기는 함 (받는 버퍼가 차지 않게)
            with lock:
                st["view"] = view
                st["info"] = {
                    "x": x,
                    "err": err,
                    "steer": steer,
                    "cmd": cmd.strip(),
                    "moving": moving,
                    "lost": lost,
                    "frames": st["frames"],
                    "note": note,
                }
                if on("arduino_status"):
                    st["info"]["status"] = status

    threading.Thread(target=loop, daemon=True).start()

    app = Flask(__name__)

    def stream():
        while True:
            with lock:
                v = st["view"]
            if v is not None:
                ok, jpg = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 75])
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            time.sleep(0.07)

    @app.route("/")
    def index():
        return PAGE.replace("__S__", json.dumps(S))

    @app.route("/stream")
    def video():
        return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api", methods=["POST"])
    def api():
        d = request.get_json(silent=True) or {}
        with lock:
            st["auto"], st["rec"], st["beat"] = (
                bool(d.get("auto")),
                bool(d.get("rec")),
                time.time(),
            )
            for k, v in (d.get("s") or {}).items():
                if k in S:
                    S[k] = float(v)
            return jsonify(st["info"])

    print(f"http://<Pi IP>:{PORT}  ·  저장 위치 {DATA_DIR}")
    try:
        app.run(host="0.0.0.0", port=PORT, threaded=True)
    finally:
        link.close()
        cam.stop()


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>cv_lab</title>
<style>
 body{margin:0;padding:12px;background:#111;color:#eee;font-family:sans-serif}
 img{width:100%;max-width:640px;border-radius:8px;background:#000;display:block}
 button{font-size:18px;padding:10px 18px;margin:8px 6px 8px 0;border:0;border-radius:8px;color:#fff;background:#1e8e3e}
 button.on{background:#d93025} #rec{background:#1a73e8} #rec.on{background:#d93025}
 .sl{display:grid;grid-template-columns:80px 1fr 50px;gap:6px;align-items:center;max-width:640px;font-size:14px}
 .info{font-family:monospace;font-size:15px;color:#f6ad55;margin:8px 0}
</style></head><body>
<h3>cv_lab: 왼쪽 = 카메라 (빨간 점 = 찾은 선 3지점, 노란 띠 = 횡단보도), 오른쪽 = 이진화</h3>
<img src="/stream">
<div><button id="go">출발 (Space)</button><button id="rec">녹화 (R)</button></div>
<div class="info" id="info">-</div>
<div class="info" id="status">-</div>
<div id="sliders"></div>
<p style="color:#999;font-size:13px">창을 닫거나 연결이 끊기면 0.5초 안에 정지 · 선을 잠깐 놓치면 하던 대로(HOLD), 6프레임 연속 놓치면 정지</p>
<script>
 const S = __S__, R = {thresh:[0,255,5], roi_top:[0,0.8,0.05], look:[0.1,0.9,0.05], kp:[0,3,0.1], ki:[0,1,0.05], kd:[0,3,0.1], speed:[0.05,0.7,0.05]};
 let auto=false, rec=false; const $=id=>document.getElementById(id);
 $('sliders').innerHTML = Object.keys(R).map(k=>`<div class="sl"><span>${k}</span><input id="${k}" type="range" min="${R[k][0]}" max="${R[k][1]}" step="${R[k][2]}" value="${S[k]}"><b id="${k}v">${S[k]}</b></div>`).join('');
 Object.keys(R).forEach(k=>$(k).oninput=()=>{S[k]=Number($(k).value); $(k+'v').textContent=S[k];});
 const render=()=>{$('go').textContent=auto?'정지 (Space)':'출발 (Space)'; $('go').classList.toggle('on',auto); $('rec').classList.toggle('on',rec); $('rec').textContent=rec?'녹화 중지 (R)':'녹화 (R)';};
 $('go').onclick=()=>{auto=!auto;render()}; $('rec').onclick=()=>{rec=!rec;render()};
 document.onkeydown=e=>{if(e.key===' '){e.preventDefault();auto=!auto;render()} if(e.key==='r'||e.key==='R'){rec=!rec;render()}};
 setInterval(async()=>{try{const r=await(await fetch('/api',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({auto,rec,s:S})})).json();
   $('info').textContent=`${r.note||''}  ·  명령 ${r.cmd||'-'}  ·  저장 ${r.frames||0}장`;
   $('status').hidden=!('status' in r); $('status').textContent=r.status||'아두이노 상태 없음';}catch(e){}},100);
 render();
</script></body></html>"""


if __name__ == "__main__":
    main()
