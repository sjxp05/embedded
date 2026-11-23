OpenCV 선 따라가기 짝 프로그래밍 실습 (1차시)
=============================================

파일
  cv_lab.py      실습 코드. [C1] ~ [C5] 함수의 TODO를 둘이 함께 작성 (분담은 자유)
  check_cv.py    카메라 · 아두이노 없이 C1 ~ C5 자동 확인 (합성 사진 사용)
  arduino/motor_serial/motor_serial.ino   아두이노 코드 (Pi의 M,왼쪽,오른쪽 명령으로 두 바퀴 제어)

준비 (Pi 5, 한 번만)
  1) 패키지 설치
     sudo apt update
     sudo apt install -y python3-venv python3-picamera2 python3-opencv python3-numpy python3-flask python3-serial
  2) 실습 폴더와 가상환경
     mkdir -p ~/codes/cv_lab && cd ~/codes/cv_lab
     unzip ~/Downloads/cv_lab_student.zip          (받은 위치에 맞게)
     python3 -m venv --system-site-packages venv   (apt로 설치한 picamera2 · OpenCV를 venv에서도 씀)
     source venv/bin/activate                      (새 터미널마다)
     python3 -c "import cv2, numpy, flask, serial, picamera2; print('OK', cv2.__version__)"
  3) 아두이노: arduino/motor_serial/motor_serial.ino 업로드 (PC의 아두이노 IDE에서, 115200 bps)
     핀: 왼쪽 모터 D3 · D11, 오른쪽 모터 D9 · D10   (D5 · D6은 PWM 주파수가 달라 쓰지 않음)
     파일 위쪽에서 자기 차에 맞게:
       MIN_PWM      바퀴가 돌기 시작하는 PWM (예: 125)
       RIGHT_BOOST  전진할 때 차가 오른쪽으로 치우치면 10부터 올려 보기 (왼쪽이면 LEFT_BOOST)
       LEFT_DIR / RIGHT_DIR   '전진'인데 바퀴가 뒤로 돌면 그 바퀴를 -1
     motor_lab.ino를 이미 올려 두었다면 그대로 써도 됨 (같은 M 명령)
     업로드 뒤 시리얼 모니터는 꼭 닫기 (Pi가 포트를 써야 함)
  4) 카메라 방향: cv_lab.py 위쪽 ROTATE_180 = 카메라를 거꾸로 달았으면 True
     (실행해서 화면이 뒤집혀 보이면 True ↔ False)
  5) 카메라 화각: 센서 전체를 쓰도록 되어 있음 (실행하면 '카메라: 센서 모드 (1640, 1232) → 화면 (320, 240)')
  6) 다른 프로그램 끄기: 카메라를 쓰는 다른 프로그램, 아두이노 IDE 시리얼 모니터
     (카메라와 시리얼 포트는 한 번에 한 프로그램만 쓸 수 있음)

순서
  0) cd ~/codes/cv_lab && source venv/bin/activate
  1) python3 check_cv.py            함수 하나 끝낼 때마다. 21개 모두 [통과]가 목표
     python3 check_cv.py cv_lab_A.py   (나눠서 할 때: 자기 파일만)
     [통과] 맞음 · [실패] 값이 틀림(힌트 보기) · [오류] 실행 중 에러 · [아직] 아직 작성 안 함
  2) 차를 책 위에 올려 바퀴를 띄움
     python3 cv_lab.py              → 브라우저에서 http://<Pi IP>:8000   (Pi 주소: hostname -I)
     카메라 화면에 찾은 선(빨간 점)과 조향 화살표가 맞게 나오는지, '출발'에서 바퀴 방향이 맞는지 확인
  3) 트랙 위에서 '출발' (Space).  다시 Space를 누르거나 브라우저를 닫으면 (0.5초 뒤 자동 정지)
  4) 슬라이더로 thresh · roi_top · look · kp · kd · speed 조절
  5) '녹화' (R)  →  data/<날짜_시간>_lab/images/*.jpg + labels.csv

잘 안 될 때
  - ModuleNotFoundError: venv를 켰는지 확인 (source venv/bin/activate). venv를 --system-site-packages 없이
    만들었다면 지우고 다시 만들기: rm -rf venv && python3 -m venv --system-site-packages venv
  - 카메라를 열 수 없음 (Device or resource busy): 카메라를 쓰는 다른 프로그램을 끄기
  - 아두이노 연결 안 됨: USB 케이블, 시리얼 모니터 닫기, ls /dev/ttyACM*
  - Permission denied: '/dev/ttyACM0' 일 때만:
      sudo usermod -aG dialout $USER   → 로그아웃 후 다시 로그인 (또는 재부팅)

주의
  - 선을 잠깐 놓치면 하던 조향 그대로(화면 HOLD, 이때는 녹화 안 함), 6프레임(약 0.3초) 연속 놓치면 멈춤 (LOST_STOP)
    (실습판에는 제자리 회전 회복이 없음 → 텀 프로젝트에서 추가)
  - 놓쳤다가 다시 찾은 첫 프레임은 D(변화)를 0으로 계산 (옛 오차와 비교해 조향이 튀지 않게)
  - 함수를 아직 안 쓴 상태로 실행해도 차는 움직이지 않고 화면에 "C1 아직" 같은 안내가 나옴
