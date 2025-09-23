# Jetson Orin DQN 자율주행 프로젝트

Jetson Orin에서 실행되는 DQN(Deep Q-Network) 기반 자율주행 RC카 시스템입니다.

## 프로젝트 개요

이 프로젝트는 Jetson Orin 개발보드를 사용하여 실시간 자율주행을 구현합니다. DQN 강화학습 알고리즘을 통해 차선 인식과 장애물 회피를 학습하며, 아두이노를 통해 RC카를 제어합니다.

## 주요 기능

- **DQN 강화학습**: 8개의 훈련 비디오로부터 자율주행 정책 학습
- **실시간 차선 인식**: HSV 색상 공간을 이용한 흰색/노란색 차선 검출
- **장애물 감지**: 빨간색 물체 감지를 통한 장애물 회피
- **Jetson Orin 최적화**: CUDA 메모리 최적화 및 성능 모니터링
- **아두이노 연동**: L298N 모터 드라이버를 통한 RC카 제어

## 시스템 요구사항

### 하드웨어
- NVIDIA Jetson Orin 개발보드
- See3CAM_CU27 카메라
- 아두이노 Uno/Nano
- L298N 모터 드라이버
- RC카 섀시
- 엔코더 (선택사항)

### 소프트웨어
- JetPack 5.0 이상
- Python 3.8+
- PyTorch (CUDA 11.8)
- OpenCV
- NumPy
- psutil

## 설치 방법

### 1. 필수 패키지 설치

```bash
# PyTorch 설치 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 기타 필수 패키지
pip install opencv-python numpy psutil pyserial
```

### 2. 비디오 파일 준비

훈련용 비디오 파일을 다음 경로에 배치:
- `/home/nvidia/videos/` 또는 현재 디렉토리
- 파일명: `1_1.mp4`, `1_2.mp4`, `4_1.mp4`, `4_2.mp4`, `5_1.mp4`, `5_2.mp4`, `6_1.mp4`, `6_2.mp4`

## 사용 방법

### 1. DQN 학습

```bash
# 실행 스크립트 사용
chmod +x run_jetson.sh
./run_jetson.sh

# 또는 직접 실행
python3 dqn.py
```

### 2. 자율주행 실행

```bash
python3 jetson_arduino_rc_controller.py
```

### 3. 수동 제어 모드

키보드 입력을 통한 수동 제어:
- `w`: 전진
- `s`: 후진
- `a`: 좌회전
- `d`: 우회전
- `x`: 정지
- `q`: 종료

## 파일 구조

```
├── dqn.py                           # DQN 학습 및 추론 코드
├── jetson_arduino_rc_controller.py  # 아두이노 RC카 제어 코드
├── run_jetson.sh                    # 실행 스크립트
└── README.md                        # 프로젝트 문서
```

## 주요 클래스

### ObstacleDetector
- 빨간색 물체 감지를 통한 장애물 검출
- HSV 색상 공간을 이용한 색상 필터링
- 컨투어 분석을 통한 물체 크기 계산

### LaneDetector
- 흰색/노란색 차선 검출
- Canny 엣지 검출 및 Hough 변환
- 차선 상태 분류 (좌측/중앙/우측)

### DQN
- 5차원 상태 입력 (차선 정보, 차량 위치, 장애물 정보)
- 3개 액션 출력 (좌회전/직진/우회전)
- 타겟 네트워크를 이용한 안정적 학습

## 성능 최적화

### Jetson Orin 최적화
- CUDA 메모리 사용량 모니터링 (1.5GB 제한)
- 배치 처리를 통한 메모리 효율성 향상
- OpenCV 스레드 수 제한
- 주기적 가비지 컬렉션

### 메모리 관리
- 프레임 단위 배치 처리
- GPU 메모리 사용량 실시간 모니터링
- 메모리 사용량 초과 시 자동 중단

## 문제 해결

### 일반적인 문제

1. **CUDA 메모리 부족**
   - 배치 크기 줄이기
   - 메모리 사용량 모니터링 확인

2. **비디오 파일 없음**
   - 테스트용 샘플 프레임 자동 생성
   - 실제 비디오 파일 추가 권장

3. **아두이노 연결 실패**
   - 포트 번호 확인 (`/dev/ttyUSB0`, `/dev/ttyACM0`)
   - 시리얼 통신 권한 확인

### 성능 튜닝

- GPU 온도 모니터링
- CPU 사용률 확인
- 메모리 사용량 최적화
- 프레임 처리 속도 조정

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 기여

버그 리포트나 기능 개선 제안은 이슈로 등록해 주세요.

## 참고 자료

- [PyTorch 공식 문서](https://pytorch.org/docs/)
- [OpenCV Python 튜토리얼](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Jetson Orin 개발자 가이드](https://developer.nvidia.com/embedded/jetson-orin)
