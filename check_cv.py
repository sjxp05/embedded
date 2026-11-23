#!/usr/bin/env python3
"""
check_cv.py - C1 ~ C5 + 추가 1 ~ 3 자동 확인 (카메라 · 아두이노 없이)

  python3 check_cv.py              # 같은 폴더의 cv_lab.py 확인
  python3 check_cv.py 다른파일.py

합성 사진(밝은 바닥 320 × 240 + 검은 선)으로 각 함수를 시험한다.
추가 1 ~ 3 함수가 없는 파일이면 그 부분은 건너뛴다.
추가 기능은 FEATURES 설정과 상관없이 모두 켜고 시험하고, 끈 경우는 맨 뒤에서 따로 시험한다.
(온도 감속 자체는 아두이노에서 하므로 여기서는 상태를 받아 읽는 부분만 확인)
"""
import csv
import importlib.util
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "cv_lab.py")
spec = importlib.util.spec_from_file_location("cv_lab", path)
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)

passed = failed = 0


def check(name, func):
    global passed, failed
    try:
        msg = func()
        print(f"  [통과] {name}" + (f"  {msg}" if msg else ""))
        passed += 1
    except NotImplementedError:
        print(f"  [아직] {name}  (함수를 아직 작성하지 않음)")
        failed += 1
    except AssertionError as e:
        print(f"  [실패] {name}  {e}")
        failed += 1
    except Exception as e:
        print(f"  [오류] {name}  {e.__class__.__name__}: {e}")
        failed += 1


def floor(w=320, h=240, value=200):
    img = np.full((h, w, 3), value, np.uint8)
    img[: int(h * 0.3)] = (150, 160, 170)          # 위쪽 = 벽 (어두운 편)
    return img


def line_img(x, top=100, width=12):
    img = floor()
    cv2.rectangle(img, (x - width // 2, top), (x + width // 2, 239), (25, 25, 25), -1)
    return img


# ---------------- C1 ----------------
def c1_shape():
    mask, y0 = lab.make_mask(line_img(200), 0.45, 80)
    assert y0 == int(240 * 0.45), f"y0 = {y0}, 기대 {int(240 * 0.45)} (세로 × roi_top)"
    assert mask.shape == (240 - y0, 320), f"mask 크기 {mask.shape}, 기대 {(240 - y0, 320)} (흑백 2차원)"


def c1_values():
    mask, y0 = lab.make_mask(line_img(200), 0.45, 80)
    on, off = mask[60, 200], mask[60, 50]
    assert on == 255, f"선 위치(x 200)의 값 {on}, 기대 255 (어두운 곳 → 흰색: THRESH_BINARY_INV)"
    assert off == 0, f"바닥(x 50)의 값 {off}, 기대 0"
    assert set(np.unique(mask)) <= {0, 255}, "값은 0 또는 255만"


def c1_thresh():
    img = floor(value=100)                          # 어두운 바닥
    mask, _ = lab.make_mask(img, 0.45, 80)
    assert mask.max() == 0, "바닥 밝기 100, 문턱 80 → 선이 아님(0)이어야 함"
    mask, _ = lab.make_mask(img, 0.45, 120)
    assert mask.min() == 255, "바닥 밝기 100, 문턱 120 → 모두 선(255)이어야 함 (thresh 인자를 썼는지)"


def c1_noise():
    img = line_img(200)
    img[180, 60] = (0, 0, 0)                        # 점 하나 (잡티)
    mask, y0 = lab.make_mask(img, 0.45, 80)
    assert mask[180 - y0, 60] == 0, "작은 점(1픽셀)은 지워져야 함 (흐리게 + 열림 연산)"


print("C1  make_mask(bgr, roi_top, thresh)")
check("반환 (mask, y0) · 크기", c1_shape)
check("선 = 255, 바닥 = 0", c1_values)
check("문턱 thresh 적용", c1_thresh)
check("작은 잡티 제거", c1_noise)


# ---------------- C2 ----------------
def c2_case(img, look, want, tol=3):
    def f():
        mask, _ = lab.make_mask(img, 0.45, 80) if img is not None else (np.zeros((132, 320), np.uint8), 0)
        got = lab.find_line_x(mask, look)
        if want is None:
            assert got is None, f"선이 없으면 None, 받은 값 {got}"
        else:
            assert got is not None, "선이 있는데 None"
            assert abs(got - want) <= tol, f"x = {got:.1f}, 기대 약 {want}"
    return f


bent = floor()
cv2.line(bent, (160, 239), (250, 110), (25, 25, 25), 12)     # 오른쪽으로 기운 선
print("\nC2  find_line_x(mask, look)")
check("선 x 200", c2_case(line_img(200), 0.6, 200))
check("선 x 60", c2_case(line_img(60), 0.6, 60))
check("선 없음 → None", c2_case(None, 0.6, None))
check("띠 위치 look: 기운 선", c2_case(bent, 0.2, 228, tol=8))
check("띠 위치 look: 기운 선 (아래)", c2_case(bent, 0.9, 170, tol=8))
short = line_img(200, top=200)                                 # 아래쪽에만 있는 짧은 선
check("띠 밖의 선은 무시 → None", c2_case(short, 0.3, None))


# ---------------- C3 ----------------
def c3_case(x, want):
    def f():
        got = lab.line_error(x, 320)
        assert abs(got - want) < 1e-6, f"line_error({x}, 320) = {got}, 기대 {want}"
    return f


print("\nC3  line_error(x, width)")
check("가운데 x 160 → 0", c3_case(160, 0.0))
check("오른쪽 x 240 → +0.5", c3_case(240, 0.5))
check("왼쪽 끝 x 0 → −1", c3_case(0, -1.0))
check("오른쪽 끝 x 320 → +1", c3_case(320, 1.0))


# ---------------- C4 ----------------
def c4_case(args, want):
    def f():
        got = lab.pd_steer(*args)
        assert abs(got - want) < 1e-6, f"pd_steer{args} = {got:.4f}, 기대 {want}"
    return f


print("\nC4  pd_steer(error, prev_error, dt, kp, kd)")
check("변화 없음          (0.5, 0.5, 0.05, 1, 0.5)", c4_case((0.5, 0.5, 0.05, 1.0, 0.5), 0.5))
check("오차 커짐          (0.5, 0.4, 0.1, 1, 0.5)", c4_case((0.5, 0.4, 0.1, 1.0, 0.5), 0.55))
check("0.1초 기준 변화     (0.5, 0.4, 0.05, 1, 0.5)", c4_case((0.5, 0.4, 0.05, 1.0, 0.5), 0.6))
check("왼쪽               (-0.3, -0.3, 0.05, 2, 0.5)", c4_case((-0.3, -0.3, 0.05, 2.0, 0.5), -0.6))
check("1 넘으면 자르기     (0.8, 0.0, 0.05, 2, 0.5)", c4_case((0.8, 0.0, 0.05, 2.0, 0.5), 1.0))
check("dt 0 → 변화 0       (0.5, 0.0, 0, 1, 0.5)", c4_case((0.5, 0.0, 0.0, 1.0, 0.5), 0.5))


# ---------------- C5 ----------------
def c5_case():
    folder = tempfile.mkdtemp()
    try:
        frame = line_img(200)
        n1 = lab.save_sample(folder, 1, frame, 0.25, 0.35)
        n2 = lab.save_sample(folder, 2, frame, -0.5, 0.35)
        assert n1 == "images/000001.jpg", f"반환 {n1!r}, 기대 'images/000001.jpg'"
        assert os.path.isfile(os.path.join(folder, n2)), f"사진 파일이 없음: {n2}"
        img = cv2.imread(os.path.join(folder, n1))
        assert img is not None and img.shape == frame.shape, "저장한 사진을 읽을 수 없거나 크기가 다름"
        rows = list(csv.reader(open(os.path.join(folder, "labels.csv"))))
        assert rows[0] == ["image", "steering", "throttle", "time"], f"머리줄 {rows[0]}"
        assert len(rows) == 3, f"머리줄 1 + 데이터 2줄이어야 함, 받은 줄 수 {len(rows)} (덮어쓰지 말고 추가)"
        assert rows[1][0] == "images/000001.jpg" and abs(float(rows[1][1]) - 0.25) < 1e-6, f"첫 줄 {rows[1]}"
        assert abs(float(rows[2][1]) + 0.5) < 1e-6, f"둘째 줄 조향 {rows[2][1]}, 기대 -0.5"
        return "(사진 2장, labels.csv 3줄)"
    finally:
        shutil.rmtree(folder, ignore_errors=True)


print("\nC5  save_sample(folder, index, frame, steering, throttle)")
check("사진 저장 + labels.csv 추가", c5_case)

# ---------------- 추가 기능: FEATURES와 상관없이 모두 켜고 시험 ----------------
SAVED_FEATURES = getattr(lab, "FEATURES", None)
if SAVED_FEATURES is not None:
    lab.FEATURES = list(lab.ALL_FEATURES)


# ---------------- 추가 1: 아두이노 상태 받기 ----------------
class FakeSerial:
    """조각난 글자를 차례로 주는 가짜 시리얼 포트"""

    def __init__(self, *chunks):
        self.chunks = list(chunks)

    @property
    def in_waiting(self):
        return len(self.chunks[0]) if self.chunks else 0

    def read(self, n):
        return self.chunks.pop(0)


def read_all(*chunks):
    link = lab.MotorLink.__new__(lab.MotorLink)      # 포트를 열지 않고 만들기
    link.ser, link.rx, link.status = FakeSerial(*chunks), "", ""
    for _ in range(len(chunks) + 1):                 # 마지막은 받은 것이 없을 때
        got = link.read_status()
    return got


def a1_case(chunks, want):
    def f():
        got = read_all(*chunks)
        assert got == want, f"받은 값 {got!r}, 기대 {want!r}"
    return f


if hasattr(lab.MotorLink, "read_status"):
    print("\n추가 1  MotorLink.read_status()")
    check("한 줄 받기", a1_case([b"TEMP=36.0C STATE=NORMAL\r\n"], "TEMP=36.0C STATE=NORMAL"))
    check("나뉘어 온 글자 잇기", a1_case([b"TEMP=36", b".0C STATE", b"=SLOW\r\n"], "TEMP=36.0C STATE=SLOW"))
    check("여러 줄이면 마지막 상태", a1_case([b"TEMP=36.0C\r\nTEMP=41.0C\r\n"], "TEMP=41.0C"))
    check("다른 줄 무시", a1_case([b"READY\r\nTEMP=36.0C\r\nPONG\r\n"], "TEMP=36.0C"))
    check("깨진 글자가 와도 오류 없음", a1_case([b"\xff\xfe\r\nTEMP=36.0C\r\n"], "TEMP=36.0C"))
else:
    print("\n추가 1  건너뜀 (read_status 없음)")


# ---------------- 추가 2: 3개 지점 + 횡단보도 ----------------
LOOK = 0.6
DT = 0.05                                            # 20fps


def points_of(img):
    mask, _ = lab.make_mask(img, 0.45, 80)
    return lab.find_line_points(mask, LOOK)


def crosswalk_img(top=100, bottom=239):
    img = line_img(160)
    img[top:bottom + 1] = 200                        # 횡단보도 안에는 길 선이 없음
    for x in range(10, 320, 40):
        cv2.rectangle(img, (x, top), (x + 18, bottom), (25, 25, 25), -1)
    return img


def a2_straight():
    xs = [x for _, x, _ in points_of(line_img(200))]
    assert len(xs) == 3, f"지점 {len(xs)}개, 기대 3개"
    assert all(x is not None and abs(x - 200) <= 3 for x in xs), f"x = {xs}, 기대 모두 약 200"


def a2_bent():
    far, mid, near = [x for _, x, _ in points_of(bent)]
    assert None not in (far, mid, near), f"x = {(far, mid, near)}, 모두 찾아야 함"
    assert far > mid > near, f"오른쪽으로 기운 선: 먼 곳 {far:.0f} > 가운데 {mid:.0f} > 가까운 곳 {near:.0f} 이어야 함"


def a2_none():
    pts = points_of(floor())
    assert all(x is None for _, x, _ in pts), f"선이 없는데 {pts}"
    assert lab.combine_points(pts) is None, "찾은 지점이 없으면 combine_points는 None"


def a2_combine():
    w = lab.LOOK_WEIGHTS
    got = lab.combine_points([(0, 100.0, 1), (0, 160.0, 1), (0, 280.0, 1)])
    want = (100 * w[0] + 160 * w[1] + 280 * w[2]) / sum(w)
    assert abs(got - want) < 1e-6, f"{got:.2f}, 기대 {want:.2f}"
    got = lab.combine_points([(0, 100.0, 1), (0, None, 0), (0, 280.0, 1)])
    want = (100 * w[0] + 280 * w[2]) / (w[0] + w[2])
    assert abs(got - want) < 1e-6, f"못 찾은 지점은 빼야 함: {got:.2f}, 기대 {want:.2f}"


def a2_cross():
    pts = points_of(crosswalk_img())
    assert all(n >= lab.CROSSWALK_MIN_LINES for _, _, n in pts), f"선 개수 {[n for _, _, n in pts]}"
    assert lab.is_crosswalk(pts), "횡단보도로 인식해야 함"
    assert lab.combine_points(pts) is None, "횡단보도 무늬는 조향에 쓰지 않아야 함 (x = None)"


def a2_cross_far():
    center = 108 + int(132 * max(0.08, min(0.92, LOOK + lab.LOOK_OFFSETS[0])))
    pts = points_of(crosswalk_img(center - 14, center + 14))
    assert pts[0][1] is None and pts[0][2] >= lab.CROSSWALK_MIN_LINES, f"먼 곳 {pts[0]}, 기대 x None"
    assert pts[2][1] is not None and abs(pts[2][1] - 160) <= 3, f"가까운 곳 x = {pts[2][1]}, 기대 약 160"
    want = lab.CROSSWALK_MIN_POINTS <= 1
    assert lab.is_crosswalk(pts) == want, f"한 지점에서만 보임 → {want} (CROSSWALK_MIN_POINTS = {lab.CROSSWALK_MIN_POINTS})"


def a2_not_cross(img):
    def f():
        pts = points_of(img)
        assert not lab.is_crosswalk(pts), f"횡단보도가 아닌데 인식함, 선 개수 {[n for _, _, n in pts]}"
    return f


def drive_cw(cw, seen, t, seconds):
    """seconds 동안 같은 장면을 보여 줌. 반환: (정지한 시간, 새 시각)"""
    stopped = 0
    for _ in range(int(round(seconds / DT))):
        stopped += lab.crosswalk_stop(cw, seen, t)
        t += DT
    return stopped * DT, t


def a2_short():
    cw = lab.new_crosswalk_state()
    t = 100.0
    for _ in range(3):                               # (FRAMES − 1)장 보였다가 사라지기를 반복
        for _ in range(lab.CROSSWALK_FRAMES - 1):
            assert not lab.crosswalk_stop(cw, True, t), f"{lab.CROSSWALK_FRAMES}프레임 연속 전에 멈춤"
            t += DT
        assert not lab.crosswalk_stop(cw, False, t), "무늬가 사라졌는데 멈춤"
        t += DT


def a2_stop_time():
    want = lab.CROSSWALK_STOP_SEC
    stopped, _ = drive_cw(lab.new_crosswalk_state(), True, 100.0, want + 5)
    assert abs(stopped - want) <= DT * 1.5, f"정지 {stopped:.2f}초, 기대 {want}초 (한 번만)"
    return f"({stopped:.2f}초)"


def a2_again():
    cw, t = lab.new_crosswalk_state(), 100.0
    want = lab.CROSSWALK_STOP_SEC
    _, t = drive_cw(cw, True, t, want + 2)           # 첫 횡단보도: 정지 후 통과 중
    s1, t = drive_cw(cw, False, t, lab.CROSSWALK_CLEAR_SEC * 0.5)
    s2, t = drive_cw(cw, True, t, 1.0)               # 잠깐 안 보였다 다시 보임 = 같은 횡단보도
    assert s1 + s2 == 0, "같은 횡단보도에서 다시 멈춤"
    _, t = drive_cw(cw, False, t, lab.CROSSWALK_CLEAR_SEC + 1)
    s3, t = drive_cw(cw, True, t, want + 2)          # 다음 횡단보도
    assert abs(s3 - want) <= DT * 1.5, f"다음 횡단보도 정지 {s3:.2f}초, 기대 {want}초"


if hasattr(lab, "crosswalk_stop"):
    two_lines = line_img(100)
    cv2.rectangle(two_lines, (220, 100), (232, 239), (25, 25, 25), -1)
    stop_line = floor()
    stop_line[150:] = 25
    print("\n추가 2  find_line_points(mask, look) · crosswalk_stop(cw, seen, now)")
    check("3개 지점: 곧은 선 x 200", a2_straight)
    check("3개 지점: 기운 선", a2_bent)
    check("선 없음 → None", a2_none)
    check("combine_points 가중 평균", a2_combine)
    check("횡단보도 인식", a2_cross)
    check("먼 곳에만 횡단보도", a2_cross_far)
    check("선 1개는 횡단보도 아님", a2_not_cross(line_img(160)))
    check("선 2개는 횡단보도 아님", a2_not_cross(two_lines))
    check("가로로 긴 선은 횡단보도 아님", a2_not_cross(stop_line))
    check("잠깐 보인 무늬는 무시", a2_short)
    check("정지 시간 CROSSWALK_STOP_SEC", a2_stop_time)
    check("같은 곳은 통과, 다음 횡단보도에서 다시 정지", a2_again)
else:
    print("\n추가 2  건너뜀 (crosswalk_stop 없음)")


# ---------------- 추가 3: PID ----------------
def near(got, want, what):
    assert abs(got - want) < 1e-6, f"{what} {got:.4f}, 기대 {want:.4f}"


def with_filter(sec, func):
    """D_FILTER_SEC를 잠깐 바꿔서 시험 (끝나면 원래 값으로)"""
    def f():
        saved = lab.D_FILTER_SEC
        lab.D_FILTER_SEC = sec
        try:
            return func()
        finally:
            lab.D_FILTER_SEC = saved
    return f


def a3_same_as_pd():
    for error, prev, dt, kp, kd in [(0.5, 0.5, 0.05, 1, 0.5), (0.5, 0.4, 0.1, 1, 0.5), (0.5, 0.4, 0.05, 1, 0.5),
                                    (-0.3, -0.3, 0.05, 2, 0.5), (0.8, 0.0, 0.05, 2, 0.5), (0.5, 0.0, 0.0, 1, 0.5)]:
        pid = lab.new_pid_state()
        pid["prev"] = prev
        near(lab.pid_steer(pid, error, dt, kp, 0, kd), lab.pd_steer(error, prev, dt, kp, kd), f"오차 {error}, 이전 {prev}: ")


def a3_i():
    pid = lab.new_pid_state()
    ki = lab.I_LIMIT                                 # 1초 동안 오차 0.5 → I_LIMIT의 절반
    for _ in range(20):
        got = lab.pid_steer(pid, 0.5, DT, 1.0, ki, 0)
    near(got, 0.5 + ki * 0.5, "1초 뒤 조향")
    pid = lab.new_pid_state()
    for _ in range(20):
        got = lab.pid_steer(pid, -0.5, DT, 1.0, ki, 0)
    near(got, -0.5 - ki * 0.5, "1초 뒤 조향 (왼쪽)")


def a3_limit():
    for sign in (1, -1):
        pid = lab.new_pid_state()
        for _ in range(2000):
            lab.pid_steer(pid, sign * 0.2, DT, 1.0, 1.0, 0)
        near(pid["i"], sign * lab.I_LIMIT, "오래 쌓은 I")


def a3_saturated():
    pid = lab.new_pid_state()
    for _ in range(100):
        lab.pid_steer(pid, 1.0, DT, 2.0, 0.5, 0)
    near(pid["i"], 0.0, "조향이 끝까지 꺾인 동안의 I")
    pid = lab.new_pid_state()
    pid["i"], pid["prev"] = lab.I_LIMIT, -1.0        # D 때문에 오른쪽 끝까지 꺾였지만 오차는 왼쪽
    lab.pid_steer(pid, -0.2, DT, 1.0, 0.5, 10.0)
    assert pid["i"] < lab.I_LIMIT, "오차가 조향과 반대 방향이면 쌓인 I가 줄어야 함"


def a3_paused():
    pid = lab.new_pid_state()
    for _ in range(100):
        lab.pid_steer(pid, 0.3, DT, 1.0, 0.5, 0, accumulate=False)
    near(pid["i"], 0.0, "정지 중의 I")


def a3_ki_zero():
    pid = lab.new_pid_state()
    pid["i"] = 0.2
    near(lab.pid_steer(pid, 0.1, DT, 1.0, 0, 0), 0.1, "ki 0일 때 조향")
    near(pid["i"], 0.0, "ki 0일 때 I")


def a3_found_again():
    pid = lab.new_pid_state()
    lab.pid_steer(pid, 0.2, DT, 1.0, 0.1, 0.5)
    i = pid["i"]
    near(lab.pid_steer(pid, 0.6, 0.0, 1.0, 0.1, 0.5), 0.6 + i, "dt 0일 때 조향 (D 없음, I 그대로)")
    near(lab.pid_steer(pid, 0.6, DT, 1.0, 0, 0.5), 0.6, "그 다음 프레임 (옛 오차와 비교하지 않음)")


def a3_filter():
    def jitter():
        pid, out = lab.new_pid_state(), []
        for k in range(100):                         # 오차가 프레임마다 ±0.02로 떨림
            out.append(lab.pid_steer(pid, 0.02 * (-1) ** k, DT, 1.0, 0, 0.5))
        return np.abs(np.diff(out[10:])).mean()
    raw = with_filter(0.0, jitter)()
    smooth = with_filter(lab.D_FILTER_SEC or 0.05, jitter)()
    assert smooth < raw, f"떨림 {smooth:.3f}, 거르기 전 {raw:.3f}보다 작아야 함"
    return f"(떨림 {raw:.3f} → {smooth:.3f})"


def a3_clip():
    for sign in (1, -1):
        near(lab.pid_steer(lab.new_pid_state(), sign * 0.9, DT, 3.0, 0.5, 3.0), sign * 1.0, "조향")


if hasattr(lab, "pid_steer"):
    print("\n추가 3  pid_steer(pid, error, dt, kp, ki, kd)")
    check("ki 0 · 필터 0 이면 pd_steer와 같음", with_filter(0.0, a3_same_as_pd))
    check("I: 시간에 따라 쌓임", a3_i)
    check("I: ±I_LIMIT까지만", a3_limit)
    check("I: 조향이 끝까지 꺾이면 쌓지 않음", a3_saturated)
    check("I: 차가 멈춘 동안 쌓지 않음", a3_paused)
    check("I: ki 0 이면 지움", a3_ki_zero)
    check("D: 선을 다시 찾은 첫 프레임은 0", a3_found_again)
    check("D: 잡음 거르기", a3_filter)
    check("−1 ~ 1로 자르기", a3_clip)
else:
    print("\n추가 3  건너뜀 (pid_steer 없음)")


# ---------------- 기능을 껐을 때 ----------------
def without(*names):
    """names만 끈 상태로 시험"""
    def wrap(func):
        def f():
            lab.FEATURES = [n for n in lab.ALL_FEATURES if n not in names]
            try:
                return func()
            finally:
                lab.FEATURES = list(lab.ALL_FEATURES)
        return f
    return wrap


def same_as_find_line_x(img):
    mask, _ = lab.make_mask(img, 0.45, 80)
    pts = lab.find_line_points(mask, LOOK)
    old = lab.find_line_x(mask, LOOK)
    new = lab.combine_points(pts)
    assert (old is None) == (new is None) and (old is None or abs(old - new) < 1e-9), f"x = {new}, 기존 find_line_x {old}"
    return pts


@without("three_points")
def a4_one_point():
    pts = same_as_find_line_x(bent)
    assert len(pts) == 1 and abs(pts[0][0] - LOOK) < 1e-9, f"지점 {[round(p[0], 2) for p in pts]}, 기대 look {LOOK} 하나"
    assert lab.is_crosswalk(points_of(crosswalk_img())), "1지점이어도 횡단보도는 인식해야 함"


@without("crosswalk")
def a4_no_crosswalk():
    pts = points_of(crosswalk_img())
    assert not lab.is_crosswalk(pts), "꺼져 있으면 횡단보도로 인식하지 않아야 함"
    assert all(x is not None for _, x, _ in pts), "꺼져 있으면 선이 여러 개여도 x를 내야 함 (기존처럼 평균)"


@without("three_points", "crosswalk")
def a4_original():
    for img in (line_img(200), line_img(60), bent, crosswalk_img(), floor(), short):
        same_as_find_line_x(img)


def steer_seq(errors, dts, kp=1.0, ki=0.3, kd=0.5):
    pid, s = lab.new_pid_state(), {"kp": kp, "ki": ki, "kd": kd}
    return [lab.compute_steer(pid, e, dt, s, True) for e, dt in zip(errors, dts)]


ERRS, DTS = [0.1, 0.3, 0.2, -0.4, -0.1, 0.5], [0.0, 0.05, 0.05, 0.1, 0.05, 0.0]


@without("pid")
def a4_pd():
    got, prev, want = steer_seq(ERRS, DTS), 0.0, []
    for e, dt in zip(ERRS, DTS):
        want.append(lab.pd_steer(e, prev, dt, 1.0, 0.5))
        prev = e
    assert all(abs(a - b) < 1e-9 for a, b in zip(got, want)), f"{got} 기대 pd_steer {want}"


def a4_pid_on():
    pid, want = lab.new_pid_state(), []
    for e, dt in zip(ERRS, DTS):
        want.append(lab.pid_steer(pid, e, dt, 1.0, 0.3, 0.5))
    got = steer_seq(ERRS, DTS)
    assert all(abs(a - b) < 1e-9 for a, b in zip(got, want)), f"{got} 기대 pid_steer {want}"


def a4_typo():
    lab.check_features(list(lab.ALL_FEATURES))
    try:
        lab.check_features(["three_point"])
    except ValueError:
        return
    raise AssertionError("오타(three_point)를 알려 주지 않음")


if SAVED_FEATURES is not None:
    print(f"\n기능 끄기  (지금 cv_lab.py에서 켠 기능: {', '.join(SAVED_FEATURES) or '없음'})")
    check("three_points 끔 → 1지점, 기존과 같은 x", a4_one_point)
    check("crosswalk 끔 → 횡단보도 무시", a4_no_crosswalk)
    check("둘 다 끔 → 기존 find_line_x와 같음", a4_original)
    check("pid 끔 → 기존 pd_steer", a4_pd)
    check("pid 켬 → pid_steer", a4_pid_on)
    check("FEATURES 오타 알림", a4_typo)
    lab.FEATURES = SAVED_FEATURES
else:
    print("\n기능 끄기  건너뜀 (FEATURES 없음)")


print(f"\n결과: 통과 {passed} / {passed + failed}")
if failed == 0:
    print("모두 통과! 바퀴를 띄우고  python3 cv_lab.py  로 화면을 확인한 뒤, 트랙 위에서 출발하세요.")
sys.exit(0 if failed == 0 else 1)
