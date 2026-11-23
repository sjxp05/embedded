/*
 * motor_serial.ino - Raspberry Pi의 시리얼 명령으로 바퀴 2개 제어 (L9110S 모터 드라이버, cv_lab · linecar 공용)
 *   + 온습도 센서(DHT11 / DHT22)로 모터 온도를 재서 기준을 넘으면 감속, 상태를 시리얼로 출력
 *
 * 추가 배선
 *   DHT    VCC → 5V,  GND → GND,  DATA → D2   (인터럽트를 쓰므로 D2. 라이브러리 설치는 필요 없음)
 *   배터리  (+) ─ R1 ─┬─ R2 ─ GND,  ┬ → A0     (선택. 저항 없이 직접 연결하면 보드가 망가짐)
 */

// ================= 기능 켜기 / 끄기 =================
// 끄려면 그 줄 앞에 // 를 붙여 주석 처리
// #define USE_TEMP_SLOWDOWN    // [추가 1] 온도가 TEMP_LIMIT_C를 넘으면 감속
// #define USE_STATUS_PRINT     // [추가 1] 온도 · 습도 · 속도 · 배터리를 시리얼로 출력
// #define USE_BATTERY       // [추가 1] 배터리 잔량 측정 (분압 회로를 A0에 연결한 뒤 켜기)

// ================= 설정 (온도 · 배터리) =================
const unsigned long TEMP_INTERVAL_MS = 100;    // 온도를 확인하는 간격 (100 = 0.1초)
const unsigned long PRINT_INTERVAL_MS = 500;   // 시리얼 모니터에 출력하는 간격
const float TEMP_LIMIT_C = 40.0;               // 기준 온도: 이보다 높으면 감속 (DHT11은 50도까지만 측정)
const float TEMP_HYST_C = 2.0;                 // 기준보다 이만큼 식어야 원래 속도로 (감속 · 해제 반복 방지)
const float HOT_SPEED_SCALE = 0.6;             // 과열 시 속도 배율 (0.6 = 60%)

const int DHT_PIN = 2;
const int DHT_TYPE = 11;                       // 11 = DHT11, 22 = DHT22
// 센서가 새 값을 만드는 데 걸리는 최소 시간. 이보다 자주 읽으면 응답하지 않으므로 줄이지 말 것
const unsigned long DHT_MIN_INTERVAL_MS = (DHT_TYPE == 22) ? 2000 : 1000;
const unsigned long DHT_LOST_MS = 5000;        // 이 시간 동안 못 읽으면 TEMP=ERR

const int BATTERY_PIN = A0;                    // 분압 회로를 연결한 핀 (USE_BATTERY를 켰을 때만 사용)
const float BATTERY_R1 = 20000.0;              // 배터리 (+) 쪽 저항 (옴)
const float BATTERY_R2 = 10000.0;              // GND 쪽 저항 (옴)
const float BATTERY_FULL_V = 8.4;              // 100% 전압 (예: 리튬이온 2개 직렬 8.4)
const float BATTERY_EMPTY_V = 6.4;             // 0% 전압

const int L_IA = 3,  L_IB = 11;   // 왼쪽 바퀴  (L9110S A-IA, A-IB)
const int R_IA = 9,  R_IB = 10;   // 오른쪽 바퀴 (L9110S B-IA, B-IB)

const unsigned long TIMEOUT_MS = 500;   // 명령이 끊기면 정지할 시간
const int MIN_PWM = 125;                // 이보다 작으면 모터가 돌지 않음

const int LEFT_BOOST = 0;
const int RIGHT_BOOST = 0;             // 전진 시 오른쪽 모터 보정 (10부터 시작해 튜닝)

const int LEFT_DIR = 1;
const int RIGHT_DIR = 1;

char buf[32];
int len = 0;
unsigned long lastCmd = 0;
bool stopped = true;

int outLeft = 0, outRight = 0;         // 감속까지 반영한 현재 명령 (-255 ~ 255)
bool overheated = false;               // true면 감속 중
float tempC = 0, humidity = 0;
bool dhtEverRead = false;
unsigned long lastDhtOk = 0;

// DHT 신호: 하강 에지 사이의 시간(us)을 인터럽트로 기록
volatile uint8_t dhtGap[44];
volatile uint8_t dhtEdges = 0;
volatile unsigned long dhtLastEdge = 0;

void setMotor(int ia, int ib, int speed, int boost) {
  speed = constrain(speed, -255, 255);
  int pwm = abs(speed);

  // 전진(speed > 0)일 때만 boost 적용, 후진은 기존 MIN_PWM 유지
  int appliedBoost = (speed > 0) ? boost : 0;
  if (pwm > 0) {
    pwm = map(pwm, 1, 255, constrain(MIN_PWM + appliedBoost, 0, 255), 255);
  }

  if (speed > 0) {          // 전진
    analogWrite(ia, pwm);
    digitalWrite(ib, LOW);
  } else if (speed < 0) {   // 후진
    digitalWrite(ia, LOW);
    analogWrite(ib, pwm);
  } else {                  // 정지
    digitalWrite(ia, LOW);
    digitalWrite(ib, LOW);
  }
}

void drive(int left, int right) {
  if (overheated) {                    // 좌우 비율(조향)은 그대로 두고 속도만 줄임
    left = (int)(left * HOT_SPEED_SCALE);
    right = (int)(right * HOT_SPEED_SCALE);
  }
  outLeft = left;
  outRight = right;
  setMotor(L_IA, L_IB, left * LEFT_DIR, LEFT_BOOST);
  setMotor(R_IA, R_IB, right * RIGHT_DIR, RIGHT_BOOST);
  stopped = (left == 0 && right == 0);
}

// 모터 테스트용: 변환 없이 PWM 값 그대로 두 바퀴 전진
void rawForward(int pwm) {
  pwm = constrain(pwm, 0, 255);
  digitalWrite(L_IB, LOW);  analogWrite(L_IA, pwm);
  digitalWrite(R_IB, LOW);  analogWrite(R_IA, pwm);
  outLeft = outRight = pwm;
  stopped = (pwm == 0);
}

// ---------------------------------------------------------------------
// DHT 읽기: 기다리지 않고(delay 없이) 조금씩 진행.
// 인터럽트를 막지 않으므로 읽는 중에도 Pi의 모터 명령을 놓치지 않음
// ---------------------------------------------------------------------
void dhtEdge() {
  unsigned long now = micros();
  unsigned long gap = now - dhtLastEdge;
  dhtLastEdge = now;
  if (dhtEdges < sizeof(dhtGap)) dhtGap[dhtEdges++] = (gap > 255) ? 255 : gap;
}

// 마지막 40개 간격 = 데이터 40비트 (짧으면 0: 약 77us, 길면 1: 약 120us)
bool decodeDht() {
  uint8_t n = dhtEdges;
  if (n < 40) return false;
  uint8_t d[5] = {0, 0, 0, 0, 0};
  for (uint8_t i = 0; i < 40; i++) {
    d[i / 8] = (d[i / 8] << 1) | (dhtGap[n - 40 + i] > 100 ? 1 : 0);
  }
  if ((uint8_t)(d[0] + d[1] + d[2] + d[3]) != d[4]) return false;   // 체크섬
  if (d[0] == 0 && d[1] == 0) return false;                         // 잡음만 들어온 경우

  if (DHT_TYPE == 22) {
    humidity = (d[0] * 256 + d[1]) * 0.1;
    tempC = ((d[2] & 0x7F) * 256 + d[3]) * 0.1;
    if (d[2] & 0x80) tempC = -tempC;
  } else {
    humidity = d[0] + d[1] * 0.1;
    tempC = d[2] + (d[3] & 0x7F) * 0.1;
    if (d[3] & 0x80) tempC = -tempC;
  }
  return true;
}

void updateDht() {
  static uint8_t step = 0;             // 0 대기, 1 시작 신호, 2 응답 받는 중
  static unsigned long since = 0;
  unsigned long now = millis();

  if (step == 0 && now - since >= max(TEMP_INTERVAL_MS, DHT_MIN_INTERVAL_MS)) {
    pinMode(DHT_PIN, OUTPUT);
    digitalWrite(DHT_PIN, LOW);
    since = now;
    step = 1;
  } else if (step == 1 && now - since >= 20) {
    dhtEdges = 0;
    pinMode(DHT_PIN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(DHT_PIN), dhtEdge, FALLING);
    step = 2;
  } else if (step == 2 && now - since >= 35) {
    detachInterrupt(digitalPinToInterrupt(DHT_PIN));
    if (decodeDht()) {
      dhtEverRead = true;
      lastDhtOk = now;
    }
    step = 0;
  }
}

bool tempValid() {
  return dhtEverRead && millis() - lastDhtOk < DHT_LOST_MS;
}

// ---------------------------------------------------------------------
// [추가 1] 모터 온도 확인 → 기준 초과 시 감속 (실제 감속은 drive에서)
//   센서를 못 읽는 동안에는 마지막 상태(정상 / 감속)를 유지
// ---------------------------------------------------------------------
void checkMotorTemp() {
  static unsigned long last = 0;
  if (millis() - last < TEMP_INTERVAL_MS) return;
  last = millis();
  if (!tempValid()) return;

  if (!overheated && tempC > TEMP_LIMIT_C) {
    overheated = true;
  } else if (overheated && tempC <= TEMP_LIMIT_C - TEMP_HYST_C) {
    overheated = false;
  }
}

// 예: TEMP=36.0C HUM=45% SPEED=35% BATT=7.42V(51%) STATE=NORMAL
// SPEED는 실제 주행 속도가 아니라 모터에 준 명령의 크기 (엔코더가 없음)
void printStatus() {
  static unsigned long last = 0;
  if (millis() - last < PRINT_INTERVAL_MS) return;
  last = millis();

  Serial.print(F("TEMP="));
  if (tempValid()) {
    Serial.print(tempC, 1);
    Serial.print(F("C HUM="));
    Serial.print(humidity, 0);
    Serial.print('%');
  } else {
    Serial.print(F("ERR HUM=ERR"));
  }
  Serial.print(F(" SPEED="));
  Serial.print((int)((abs(outLeft) + abs(outRight)) * 100L / 510));
  Serial.print(F("% BATT="));
#ifdef USE_BATTERY
  {
    float v = analogRead(BATTERY_PIN) * 5.0 / 1023.0 * (BATTERY_R1 + BATTERY_R2) / BATTERY_R2;
    float pct = (v - BATTERY_EMPTY_V) / (BATTERY_FULL_V - BATTERY_EMPTY_V) * 100;
    Serial.print(v, 2);
    Serial.print(F("V("));
    Serial.print((int)constrain(pct, 0, 100));
    Serial.print(F("%)"));
  }
#else
  Serial.print(F("N/A"));
#endif
  Serial.print(F(" STATE="));
  Serial.println(overheated ? F("SLOW") : F("NORMAL"));
}

void handleLine(char *line) {
  int left, right, pwm;
  if (sscanf(line, "M,%d,%d", &left, &right) == 2) {
    drive(left, right);
    lastCmd = millis();
  } else if (sscanf(line, "P,%d", &pwm) == 1) {
    rawForward(pwm);
    lastCmd = millis();                 // 0.5초 뒤 자동 정지
    Serial.print("raw PWM ");
    Serial.println(pwm);
  } else if (strcmp(line, "PING") == 0) {
    Serial.println("PONG");
  }
}

void setup() {
  int pins[] = {L_IA, L_IB, R_IA, R_IB};
  for (int i = 0; i < 4; i++) pinMode(pins[i], OUTPUT);
  drive(0, 0);
  Serial.begin(115200);
  Serial.println("READY");
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      buf[len] = '\0';
      handleLine(buf);
      len = 0;
    } else if (c != '\r' && len < (int)sizeof(buf) - 1) {
      buf[len++] = c;
    }
  }

  if (!stopped && millis() - lastCmd > TIMEOUT_MS) {
    drive(0, 0);   // 명령이 끊김 → 자동 정지
  }

#if defined(USE_TEMP_SLOWDOWN) || defined(USE_STATUS_PRINT)
  updateDht();         // 감속이나 출력 중 하나라도 켜져 있으면 센서를 읽음
#endif
#ifdef USE_TEMP_SLOWDOWN
  checkMotorTemp();
#endif
#ifdef USE_STATUS_PRINT
  printStatus();
#endif
}
