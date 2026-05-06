# Preprocess Report

담당 REAL 범위: REAL07\~REAL08
처리 날짜: 2026-04-28
담당자: 서민재

## 처리 결과

처리 샘플 수: 150
실패 샘플 수: 0

## 데이터 출처

원본 zip: 13\_real\_word\_video.zip (REAL07), 14\_real\_word\_video.zip (REAL07), 15\_real\_word\_video.zip (REAL08), 16\_real\_word\_video.zip (REAL08)
추출 경로: data\\raw\\videos

## 고정 기준

layout: mediapipe\_xyz
feature: pose33 + left\_hand21 + right\_hand21
point: \[x, y, z]
feature count: 225
sequence length: 32
normalization: shoulder-center + shoulder-width scale

## 사용 스크립트

REAL\_SHARD\_WORKER\\scripts\\build\_shard\_manifest\_from\_videos.py
REAL\_SHARD\_WORKER\\scripts\\preprocess\_shard.py

## 환경

Python: 3.10
mediapipe: 0.10.13
opencv-python: 4.11.0.86
numpy: 1.26.4

## NPZ shape

(150, 32, 225)

## label 분포

가다: 40
감사: 10
괜찮다: 10
배고프다: 20
병원: 10
아프다: 10
우유: 20
자다: 30

