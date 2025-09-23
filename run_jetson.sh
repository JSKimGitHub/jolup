#!/bin/bash
# Jetson Orin DQN 자율주행 실행 스크립트

echo "Jetson Orin DQN 자율주행 시스템 시작..."

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0
export OPENCV_VIDEOIO_PRIORITY_V4L2=1

# 필요한 패키지 확인
echo "필요한 패키지 확인 중..."
python3 -c "import torch, cv2, numpy, psutil" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "필요한 패키지가 설치되지 않았습니다."
    echo "다음 명령어로 설치하세요:"
    echo "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    echo "pip install opencv-python numpy psutil"
    exit 1
fi

# CUDA 사용 가능 여부 확인
echo "CUDA 사용 가능 여부 확인 중..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 비디오 파일 확인
echo "훈련 비디오 파일 확인 중..."
video_files=("1_1.mp4" "1_2.mp4" "4_1.mp4" "4_2.mp4" "5_1.mp4" "5_2.mp4" "6_1.mp4" "6_2.mp4")
missing_files=()

for video in "${video_files[@]}"; do
    if [ ! -f "$video" ] && [ ! -f "/home/nvidia/videos/$video" ]; then
        missing_files+=("$video")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo "다음 비디오 파일이 없습니다:"
    printf '%s\n' "${missing_files[@]}"
    echo "비디오 파일이 없어도 테스트용 샘플 데이터로 실행됩니다."
fi

# 메모리 상태 확인
echo "시스템 메모리 상태 확인 중..."
free -h

# GPU 메모리 상태 확인 (nvidia-smi가 있는 경우)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU 메모리 상태:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
fi

# DQN 학습 실행
echo "DQN 학습 시작..."
python3 dqn.py

# 학습 완료 후 결과 확인
if [ $? -eq 0 ]; then
    echo "DQN 학습이 성공적으로 완료되었습니다!"
    
    # 학습된 모델 파일 확인
    if [ -f "policy_net.pth" ]; then
        echo "학습된 모델이 저장되었습니다: policy_net.pth"
    fi
    
    # 자율주행 실행 여부 확인
    read -p "자율주행을 시작하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "자율주행 시작..."
        python3 jetson_arduino_rc_controller.py
    fi
else
    echo "DQN 학습 중 오류가 발생했습니다."
    exit 1
fi

echo "시스템 종료"
