# Image-descanning-model
Project: Restoring Scanned Images to Original

(In Korean)

1. 목표
   
스캔된 이미지에서 발생하는 다양한 열화현상을 제거하는 디스캐닝 모델 학습


3. 사용 모델
   
단순 디노이징 모델 구조를 변형함

잔차 채널 어텐션 모듈의 반복 구조


5. 주요사항
- 잔차 채널 어텐션 블록 구조 모델
- 학습 데이터셋 이미지 RGB 값 분포 계산 및 데이터 정규화
- 손실함수 : L1 loss + 0.1 * perceptual mse loss (수정될 수 있음)
- 학습률 스케줄러 : custom cosineAnnealingWarmUpRestarts lr scheduler (수정될 수 있음)

  (reference : https://gaussian37.github.io/dl-pytorch-lr_scheduler/) (not mine)
- 학습 파라미터 최적화

  에폭, 배치사이즈, 학습률, 채널 개수, 층 개수, 손실함수 등 다양한 파라미터 조정 및 실험


4. 파일 설명
- lr_scheduler.py : 커스텀 학습률 스케줄러
- test.py : 테스트 이미지셋에 대한 결과 이미지 생성
- train.py : 훈련 이미지셋에 대한 학습 수행
- y_value_mae.py : YUV 색공간에서 y채널 값의 평균 절대값 오차 계산


5. 기타
- 데이터셋은 pc에 저장되어 있음
- 아직 부족한 성능이라 모델 및 학습 파라미터에 대한 추가적인 수정이 예정됨
