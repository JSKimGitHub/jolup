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

