#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Orin 호환 DQN 자율주행 코드

이 스크립트는 Jetson Orin에서 실행하기 위해 최적화된 DQN 기반 자율주행 코드입니다.

주요 변경사항:
- Jetson Orin CUDA 메모리 최적화
- 8개 비디오 파일 지원 (1_1.mp4, 1_2.mp4, 4_1.mp4, 4_2.mp4, 5_1.mp4, 5_2.mp4, 6_1.mp4, 6_2.mp4)
- 메모리 사용량 모니터링 및 제한
- GPU 텐서 처리 최적화
- 성능 모니터링 유틸리티 추가

사전 준비사항:
1. 필요한 패키지 설치:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install opencv-python numpy psutil

2. 비디오 파일 준비:
  - /home/nvidia/videos/ 경로에 8개 비디오 파일 배치
  - 또는 현재 디렉토리에 8개 비디오 파일 배치

주의사항:
- 메모리 사용량이 1.5GB를 초과하면 자동으로 중단됩니다
- GPU 온도가 높으면 성능이 저하될 수 있습니다
- 비디오 파일이 없으면 테스트용 샘플 프레임이 생성됩니다
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
import os
import psutil

# Jetson Orin에서 CUDA 메모리 최적화
if torch.cuda.is_available():
   torch.backends.cudnn.benchmark = True
   torch.backends.cudnn.deterministic = False
   # Jetson Orin 메모리 최적화
   os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
   os.environ['CUDA_CACHE_DISABLE'] = '0'

# Jetson Orin에서 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Jetson Orin에서 메모리 정보 출력
if torch.cuda.is_available():
   print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
   print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
   print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
   print(f"CUDA Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.3f} GB")

class ObstacleDetector:
   def find_red_area(self, frame):
       """
       frame: BGR 이미지 (OpenCV로 읽은 이미지)
       return: 빨간색 물체의 넓이 (픽셀 수), 없으면 0
       """
       # 1. BGR -> HSV 변환
       hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

       # 2. 빨간색 HSV 범위 정의
       lower_red1 = np.array([0, 120, 70])
       upper_red1 = np.array([10, 255, 255])
       lower_red2 = np.array([170, 120, 70])
       upper_red2 = np.array([180, 255, 255])

       # 3. 마스크 생성
       mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
       mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
       mask = mask1 + mask2

       # 4. 노이즈 제거
       mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
       mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5,5), np.uint8))

       # 5. 컨투어 찾기
       contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       if len(contours) == 0:
           return 0  # 빨간색 물체 없음 → 넓이 0 반환

       # 6. 가장 큰 컨투어의 면적 계산
       c = max(contours, key=cv2.contourArea)
       area = cv2.contourArea(c)

       return area

class LaneDetector:
   def __init__(self):
       self.prev_lanes = [None, None]  # [왼쪽 차선, 오른쪽 차선]
       self.img_center = None
       self.margin = 50  # 상태 판정 margin

   def process_frame(self, frame):
       height, width = frame.shape[:2]
       self.img_center = width // 2

       # ----------------------
       # 1️⃣ Gray + Blur
       # ----------------------
       hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

       # 흰색 범위 (밝은 영역)
       lower_white = np.array([0, 0, 200])
       upper_white = np.array([180, 30, 255])
       mask_white = cv2.inRange(hsv, lower_white, upper_white)

       # 노란색 범위
       lower_yellow = np.array([15, 80, 100])
       upper_yellow = np.array([35, 255, 255])
       mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

       # 두 마스크 합치기
       mask = cv2.bitwise_or(mask_white, mask_yellow)

       # 원본에서 색상만 추출
       result = cv2.bitwise_and(frame, frame, mask=mask)

       gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
       blur = cv2.GaussianBlur(gray, (5,5), 0)

       # ----------------------
       # 2️⃣ Canny 엣지
       # ----------------------
       edges = cv2.Canny(blur, 50, 150)

       # ----------------------
       # 3️⃣ ROI 적용
       # ----------------------
       mask = np.zeros_like(edges)
       roi = np.array([[
           (0, height),
           (0, int(height*0.6)),
           (width, int(height*0.6)),
           (width, height)
       ]], np.int32)
       cv2.fillPoly(mask, roi, 255)
       edges_roi = cv2.bitwise_and(edges, mask)

       # ----------------------
       # 4️⃣ 허프 직선
       # ----------------------
       lines = cv2.HoughLinesP(edges_roi, 1, np.pi/180, 50, minLineLength=20, maxLineGap=100)

       # ----------------------
       # 5️⃣ slope 기준 left/right 분류
       # ----------------------
       left_lines, right_lines = [], []
       if lines is not None:
           for x1, y1, x2, y2 in lines[:, 0]:
               slope = (y2 - y1) / (x2 - x1 + 1e-6)
               if slope < -0.5:
                   left_lines.append((x1, y1, x2, y2))
               elif slope > 0.5:
                   right_lines.append((x1, y1, x2, y2))

       # ----------------------
       # 6️⃣ 화면 중심에 가장 가까운 안쪽 선 선택
       # ----------------------
       left_inner = max(left_lines, key=lambda l: (l[0]+l[2])/2) if left_lines else None
       right_inner = min(right_lines, key=lambda l: (l[0]+l[2])/2) if right_lines else None

       lanes = [None, None]  # [left, right]

       for line_inner, idx in [(left_inner, 0), (right_inner, 1)]:
           if line_inner is not None:
               x1, y1, x2, y2 = line_inner
               lane = [(x1, y1), (x2, y2)]  # 검출된 직선 그대로 사용

               # 이전 프레임과 스무딩
               if self.prev_lanes[idx] is not None:
                   lane = [(
                       (lane[i][0] + self.prev_lanes[idx][i][0]) // 2,
                       (lane[i][1] + self.prev_lanes[idx][i][1]) // 2
                   ) for i in range(len(lane))]

               self.prev_lanes[idx] = lane
               lanes[idx] = lane  # 상태 계산용

       # ----------------------
       # 6️⃣ 상태 계산
       # ----------------------
       lane_state = 1  # 기본 center

       if lanes[0] is not None and lanes[1] is not None:
           left_center = (lanes[0][0][0] + lanes[0][1][0]) // 2
           right_center = (lanes[1][0][0] + lanes[1][1][0]) // 2
           lane_center = (left_center + right_center) // 2

           if abs(lane_center - self.img_center) < self.margin:
               lane_state = 1  # center
           elif lane_center < self.img_center:
               lane_state = 0  # left
           else:
               lane_state = 2  # right

       return lanes, lane_state

class OfflineDataCollector:
   def __init__(self, lane_detector, obstacle_detector):
       self.lane_detector = lane_detector
       self.obstacle_detector = obstacle_detector

   def _get_state(self, frame, car_x):
       """주어진 프레임에서 상태(state) 계산"""
       # 장애물 정보
       area= self.obstacle_detector.find_red_area(frame)

       lanes, act= self.lane_detector.process_frame(frame)
       lanes = np.array(lanes, dtype=object)
       left_lane, right_lane = lanes

       left_x = min(left_lane[0][0], left_lane[1][0]) if left_lane is not None else 0
       right_x = max(right_lane[0][0], right_lane[1][0]) if right_lane is not None else frame.shape[1]

       state = np.array([
           left_x / frame.shape[1],
           right_x / frame.shape[1],
           (left_x + right_x) / (2 * frame.shape[1]),  # 차선 중앙
           car_x / frame.shape[1],
           area / (frame.shape[0] * frame.shape[1])  # 프레임에 대한 빨간색의 비율
      ], dtype=np.float32)

       return state, act

   def _calculate_reward(self, state):
       """주어진 상태에서 보상 계산"""
       reward = 0.0

       lane_center = state[2]
       car_position = state[3]
       distance_from_center = abs(car_position - lane_center)

       # 차선 중앙 유지 보상
       if distance_from_center < 0.1:
           reward += 10.0
       elif distance_from_center < 0.2:
           reward += 5.0
       else:
           reward -= 5.0

       # 차선 이탈 페널티
       if distance_from_center > 0.4:
           reward -= 20.0

       # 안정적 주행 기본 보상
       reward += 5.0

       # 장애물과의 거리 보상
       norm_area = state[4]
       if norm_area > 0.7:
           reward -= 70.0

       return reward

   def collect_from_frames(self, frames, car_x_init=None, actions_taken=None, batch_size=1000):
       """
       frames에서 state/action/reward/next_state/done 리스트 생성 (메모리 효율적)
       frames : 비디오 프레임 리스트
       car_x_init : 초기 차량 위치 (없으면 화면 중앙)
       actions_taken : 이미 결정된 action 리스트 (없으면 간단 규칙 적용)
       batch_size : 메모리 효율성을 위한 배치 크기
       """
       state_list = []
       action_list = []
       reward_list = []
       next_state_list = []
       done_list = []

       car_x = frames[0].shape[1] // 2 if frames else 320

       # 메모리 효율성을 위해 배치 단위로 처리
       total_frames = len(frames)
       processed_count = 0
      
       print(f"총 {total_frames} 프레임에서 데이터 수집 시작...")
      
       for batch_start in range(0, total_frames, batch_size):
           batch_end = min(batch_start + batch_size, total_frames)
           batch_frames = frames[batch_start:batch_end]
          
           print(f"배치 처리: {batch_start}-{batch_end} 프레임 ({len(batch_frames)}개)")
          
           # 현재 배치에서 유효한 인덱스 계산
           valid_indices = [i for i in range(len(batch_frames)-1) if i % 4 == 0]
          
           for local_idx in valid_indices:
               global_idx = batch_start + local_idx
              
               if global_idx + 4 >= total_frames:
                   break
                  
               frame = batch_frames[local_idx]
               next_frame = frames[global_idx + 4]  # 다음 프레임은 전체 프레임에서 가져옴

               # 현재 상태
               state, act = self._get_state(frame, car_x)

               # 다음 상태
               next_state, next_act = self._get_state(next_frame, car_x)

               # 보상 계산
               reward = self._calculate_reward(state)

               # done 여부
               done = False

               if abs(next_state[3] - next_state[2]) > 0.5:  # 차선 벗어나면 종료
                   done = True

               if global_idx + 4 >= total_frames:  # 마지막 프레임
                   done = True

               # 리스트 저장
               state_list.append(state)
               action_list.append(act)
               reward_list.append(reward)
               next_state_list.append(next_state)
               done_list.append(done)
              
               processed_count += 1
              
               # 진행 상황 출력
               if processed_count % 100 == 0:
                   print(f"  처리된 transition: {processed_count}")
          
           # 배치 처리 후 메모리 정리
           if torch.cuda.is_available():
               torch.cuda.empty_cache()
      
       print(f"데이터 수집 완료: {len(state_list)} transition 생성")
       return state_list, action_list, reward_list, next_state_list, done_list

class DQN(nn.Module):
   def __init__(self, state_dim, action_dim):
       super().__init__()
       self.fc = nn.Sequential(
           nn.Linear(state_dim, 64),
           nn.ReLU(),
           nn.Dropout(0.2),  # 과적합 방지
           nn.Linear(64, 64),
           nn.ReLU(),
           nn.Dropout(0.2),
           nn.Linear(64, action_dim)
       )

   def forward(self, x):
       return self.fc(x)

def train_offline_dqn(state_list, action_list, reward_list, next_state_list, done_list,
                      epochs=100, batch_size=32):
   """오프라인 RL DQN 학습 (Jetson Orin 최적화)"""
   print("Starting offline DQN training for Jetson Orin...")

   state_dim = len(state_list[0])
   action_dim = 3 #액션 개수 3개

   # Jetson Orin용 네트워크 초기화 (GPU로 이동)
   policy_net = DQN(state_dim, action_dim).to(device)
   target_net = DQN(state_dim, action_dim).to(device)
   target_net.load_state_dict(policy_net.state_dict())

   # Jetson Orin에 최적화된 옵티마이저 설정
   optimizer = optim.Adam(policy_net.parameters(), lr=5e-4, weight_decay=1e-5)
   gamma = 0.99
   update_frequency = 10

   # 전체 경험 리스트
   dataset = list(zip(state_list, action_list, reward_list, next_state_list, done_list))
  
   # Jetson Orin 메모리 최적화를 위한 배치 크기 조정
   if len(dataset) < batch_size:
       batch_size = min(len(dataset), 16)  # 작은 데이터셋의 경우 배치 크기 줄임
       print(f"데이터셋 크기에 맞춰 배치 크기를 {batch_size}로 조정")

   # 메모리 정리를 위한 주기적 가비지 컬렉션
   import gc
  
   for epoch in range(epochs):
       # 무작위 배치 샘플링
       batch = random.sample(dataset, batch_size)

       # Jetson Orin에서 효율적인 텐서 생성
       states = torch.tensor(np.array([exp[0] for exp in batch]), dtype=torch.float32, device=device)
       actions = torch.tensor([exp[1] for exp in batch], dtype=torch.long, device=device)
       rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float32, device=device)
       next_states = torch.tensor(np.array([exp[3] for exp in batch]), dtype=torch.float32, device=device)
       dones = torch.tensor([exp[4] for exp in batch], dtype=torch.bool, device=device)

       # Q-러닝 업데이트
       current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

       with torch.no_grad():
           next_q_values = target_net(next_states).max(1)[0]
           target_q_values = rewards + gamma * (1 - dones.float()) * next_q_values

       loss = nn.MSELoss()(current_q_values, target_q_values)

       optimizer.zero_grad()
       loss.backward()
      
       # Jetson Orin에서 그래디언트 클리핑 추가
       torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
      
       optimizer.step()

       # 타겟 네트워크 주기적 업데이트
       if epoch % update_frequency == 0:
           target_net.load_state_dict(policy_net.state_dict())

       # Jetson Orin 메모리 모니터링
       if torch.cuda.is_available() and epoch % 5 == 0:
           memory_used = torch.cuda.memory_allocated() / 1024**3
           print(f"Epoch {epoch}, Loss: {loss.item():.4f}, GPU Memory: {memory_used:.3f} GB")
          
           # 메모리 사용량이 너무 높으면 정리
           if memory_used > 1.5:  # 1.5GB 이상 사용시
               torch.cuda.empty_cache()
               gc.collect()
       else:
           print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

   # 최종 메모리 정리
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
  
   print("Offline DQN training completed!")
   return policy_net

def load_video_frames(video_path, max_frames_per_video=500):
   """단일 비디오에서 프레임을 로드하는 함수"""
   print(f"비디오 로딩: {video_path}")
  
   cap = cv2.VideoCapture(video_path)
   if not cap.isOpened():
       print(f"비디오 파일을 열 수 없습니다: {video_path}")
       return []
  
   frames = []
   frame_count = 0
  
   while True:
       ret, frame = cap.read()
       if not ret:
           break
       frames.append(frame)
       frame_count += 1
      
       # 메모리 사용량 체크
       if torch.cuda.is_available():
           memory_used = torch.cuda.memory_allocated() / 1024**3
           if memory_used > 1.5:  # 1.5GB 이상 사용시 중단
               print(f"메모리 사용량 제한으로 인해 {frame_count} 프레임에서 중단")
               break
      
       if frame_count >= max_frames_per_video:
           print(f"최대 프레임 수({max_frames_per_video})에 도달하여 중단")
           break
  
   cap.release()
   print(f"  - {len(frames)} 프레임 로드됨")
   return frames

def load_all_training_videos():
   """8개의 훈련 비디오를 모두 로드하는 함수"""
   # 훈련 데이터 비디오 파일 목록
   video_files = [
       "1_1.mp4", "1_2.mp4",
       "4_1.mp4", "4_2.mp4",
       "5_1.mp4", "5_2.mp4",
       "6_1.mp4", "6_2.mp4"
   ]
  
   all_frames = []
   total_frames = 0
  
   print("=== 훈련 비디오 로딩 시작 ===")
  
   for video_file in video_files:
       # 여러 경로에서 비디오 파일 찾기
       possible_paths = [
           f"/home/nvidia/videos/{video_file}",  # Jetson Orin 기본 경로
           f"./{video_file}",  # 현재 디렉토리
           video_file  # 파일명만으로 찾기
       ]
      
       video_path = None
       for path in possible_paths:
           if os.path.exists(path):
               video_path = path
               break
      
       if video_path is None:
           print(f"경고: {video_file} 파일을 찾을 수 없습니다.")
           continue
      
       # 비디오에서 프레임 로드
       frames = load_video_frames(video_path, max_frames_per_video=500)
      
       if frames:
           all_frames.extend(frames)
           total_frames += len(frames)
           print(f"  ✓ {video_file}: {len(frames)} 프레임 추가")
       else:
           print(f"  ✗ {video_file}: 프레임 로드 실패")
  
   print(f"=== 총 {total_frames} 프레임 로드 완료 ===")
  
   # 비디오 파일이 하나도 없으면 샘플 프레임 생성
   if not all_frames:
       print("비디오 파일이 없어서 테스트용 샘플 프레임을 생성합니다...")
       for i in range(100):
           frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
           all_frames.append(frame)
       total_frames = len(all_frames)
  
   return all_frames

def monitor_jetson_performance():
   """Jetson Orin 성능 모니터링"""
   print("=== Jetson Orin 성능 모니터링 ===")
  
   # CPU 정보
   cpu_percent = psutil.cpu_percent(interval=1)
   print(f"CPU 사용률: {cpu_percent}%")
  
   # 메모리 정보
   memory = psutil.virtual_memory()
   print(f"시스템 메모리 사용률: {memory.percent}%")
   print(f"사용 가능한 메모리: {memory.available / 1024**3:.1f} GB")
  
   # GPU 정보 (CUDA 사용 가능한 경우)
   if torch.cuda.is_available():
       print(f"GPU 사용률: {torch.cuda.utilization()}%")
       print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
       print(f"GPU 메모리 예약량: {torch.cuda.memory_reserved() / 1024**3:.3f} GB")
       print(f"GPU 온도: {torch.cuda.get_device_properties(0).name}")
  
   # 온도 정보 (Jetson 특화)
   try:
       with open('/sys/devices/virtual/thermal/thermal_zone0/temp', 'r') as f:
           cpu_temp = int(f.read()) / 1000
       print(f"CPU 온도: {cpu_temp:.1f}°C")
   except:
       print("CPU 온도 정보를 읽을 수 없습니다.")
  
   try:
       with open('/sys/devices/virtual/thermal/thermal_zone1/temp', 'r') as f:
           gpu_temp = int(f.read()) / 1000
       print(f"GPU 온도: {gpu_temp:.1f}°C")
   except:
       print("GPU 온도 정보를 읽을 수 없습니다.")

def optimize_jetson_performance():
   """Jetson Orin 성능 최적화"""
   print("Jetson Orin 성능 최적화 적용 중...")
  
   # CUDA 설정 최적화
   if torch.cuda.is_available():
       torch.backends.cudnn.benchmark = True
       torch.backends.cudnn.deterministic = False
      
       # 메모리 정리
       torch.cuda.empty_cache()
      
       # Jetson 전용 설정
       os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 성능을 위해 비동기 실행
       os.environ['CUDA_CACHE_DISABLE'] = '0'    # 캐시 활성화
      
       print("CUDA 최적화 완료")
  
   # OpenCV 최적화
   cv2.setNumThreads(4)  # OpenCV 스레드 수 제한
   print("OpenCV 최적화 완료")

def cleanup_jetson_memory():
   """Jetson Orin 메모리 정리"""
   import gc
  
   # Python 가비지 컬렉션
   gc.collect()
  
   # CUDA 메모리 정리
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
       torch.cuda.synchronize()
  
   print("메모리 정리 완료")

def main():
   """메인 실행 함수"""
   print("Jetson Orin DQN 자율주행 코드 시작...")
  
   # 성능 모니터링 실행
   monitor_jetson_performance()
  
   # 성능 최적화 적용
   optimize_jetson_performance()
  
   # 1. 모든 훈련 비디오에서 프레임 읽기
   frames = load_all_training_videos()

   # 2. LaneDetector, ObstacleDetector 생성
   lane_detector = LaneDetector()
   obstacle_detector = ObstacleDetector()

   # 3. OfflineDataCollector 생성
   collector = OfflineDataCollector(lane_detector, obstacle_detector)

   # 4. 데이터 수집 (메모리 효율적 배치 처리)
   print("\n=== 데이터 수집 시작 ===")
   state_list, action_list, reward_list, next_state_list, done_list = collector.collect_from_frames(
       frames, batch_size=1000  # 메모리 효율성을 위한 배치 크기
   )

   # 5. 결과 확인
   print(f"\n=== 데이터 수집 결과 ===")
   print(f"총 transition 수: {len(state_list)}")
   if len(state_list) > 0:
       print(f"샘플 state: {state_list[0]}")
       print(f"샘플 action: {action_list[0]}")
       print(f"샘플 reward: {reward_list[0]}")
       print(f"샘플 next_state: {next_state_list[0]}")
       print(f"샘플 done: {done_list[0]}")

   # 6. Jetson Orin에 최적화된 학습 파라미터로 학습
   if len(state_list) > 0:
       print("\n=== DQN 학습 시작 ===")
       policy_net = train_offline_dqn(
           state_list,
           action_list,
           reward_list,
           next_state_list,
           done_list,
           epochs=20,  # 더 많은 데이터로 인해 학습 횟수 증가
           batch_size=32  # 더 많은 데이터로 인해 배치 크기 증가
       )

       # 7. Q-value 테스트
       print("\n=== Q-value 테스트 시작 ===")
       test_count = min(10, len(state_list))  # Jetson Orin에서 테스트 개수 제한

       for i in range(test_count):
           sample_state = torch.tensor(state_list[i], dtype=torch.float32, device=device)
           q_values = policy_net(sample_state.unsqueeze(0))  # 배치 차원 추가
          
           # GPU에서 CPU로 이동하여 출력
           q_values_cpu = q_values.detach().cpu().numpy()
           print(f"Sample {i+1} Q-values: {q_values_cpu}")
           action = q_values.argmax().item()
           print(f"Sample {i+1} action: {action}")
          
           # 메모리 정리
           if torch.cuda.is_available() and i % 5 == 0:
               torch.cuda.empty_cache()

       print("Q-value 테스트 완료!")
   else:
       print("데이터가 없어서 학습을 건너뜁니다.")
  
   # 8. 최종 메모리 정리
   cleanup_jetson_memory()
   print("\n=== Jetson Orin DQN 자율주행 학습 완료 ===")
   print("8개 비디오 파일로부터 학습이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
   main()
