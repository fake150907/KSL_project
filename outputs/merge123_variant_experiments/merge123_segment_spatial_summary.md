# merge1~3 segment/spatial offline experiment

- sample_fps: 12.0
- segment variants: base, trim_head_0.2, trim_tail_0.2, trim_both_0.2, trim_head_0.4, trim_tail_0.4, trim_both_0.4, extend_head_0.2, extend_tail_0.2, extend_both_0.2, shift_left_0.2, shift_right_0.2
- spatial variants: orig, hflip, zoom1.10, zoom1.20, zoom1.10_hflip, zoom1.20_hflip
- elapsed_sec: 291.5

## merge1

### 1. 상처 expected=상처
- baseline: pred=새똥 pred_conf=0.181 target_conf=0.035 target_rank=4 hand_frames=51
- best by expected target confidence:
  - zoom1.20 / extend_tail_0.2 [0.0, 4.6] pred=상처(0.957) target_conf=0.957 rank=1 hand=53
  - zoom1.20 / extend_both_0.2 [0.0, 4.6] pred=상처(0.957) target_conf=0.957 rank=1 hand=53
  - zoom1.20 / trim_head_0.4 [0.4, 4.4] pred=상처(0.926) target_conf=0.926 rank=1 hand=45
  - zoom1.20 / trim_head_0.2 [0.2, 4.4] pred=상처(0.921) target_conf=0.921 rank=1 hand=47
  - zoom1.20 / shift_right_0.2 [0.2, 4.6] pred=상처(0.911) target_conf=0.911 rank=1 hand=50
- strongest top1 predictions:
  - zoom1.20_hflip / trim_head_0.2 pred=녹초(0.997) target_conf=0.000
  - zoom1.20_hflip / base pred=녹초(0.996) target_conf=0.000
  - zoom1.20_hflip / extend_head_0.2 pred=녹초(0.996) target_conf=0.000

### 2. 붕대 expected=붕대
- baseline: pred=붕대 pred_conf=0.874 target_conf=0.874 target_rank=1 hand_frames=53
- best by expected target confidence:
  - zoom1.20 / trim_head_0.2 [4.6, 8.9] pred=붕대(0.970) target_conf=0.970 rank=1 hand=43
  - zoom1.10 / extend_tail_0.2 [4.4, 9.1] pred=붕대(0.965) target_conf=0.965 rank=1 hand=56
  - zoom1.20 / base [4.4, 8.9] pred=붕대(0.961) target_conf=0.961 rank=1 hand=46
  - zoom1.20 / extend_tail_0.2 [4.4, 9.1] pred=붕대(0.960) target_conf=0.960 rank=1 hand=47
  - orig / extend_tail_0.2 [4.4, 9.1] pred=붕대(0.953) target_conf=0.953 rank=1 hand=56
- strongest top1 predictions:
  - zoom1.20 / trim_head_0.2 pred=붕대(0.970) target_conf=0.970
  - zoom1.10 / extend_tail_0.2 pred=붕대(0.965) target_conf=0.965
  - zoom1.20 / base pred=붕대(0.961) target_conf=0.961

### 3. 원하다 expected=원하다
- baseline: pred=혈서 pred_conf=0.213 target_conf=0.034 target_rank=7 hand_frames=25
- best by expected target confidence:
  - zoom1.20 / shift_left_0.2 [8.7, 10.8] pred=원하다(0.420) target_conf=0.420 rank=1 hand=21
  - zoom1.20 / extend_head_0.2 [8.7, 10.967] pred=원하다(0.411) target_conf=0.411 rank=1 hand=23
  - zoom1.20 / extend_both_0.2 [8.7, 10.967] pred=원하다(0.411) target_conf=0.411 rank=1 hand=23
  - zoom1.20 / base [8.9, 10.967] pred=원하다(0.307) target_conf=0.307 rank=1 hand=22
  - zoom1.20 / extend_tail_0.2 [8.9, 10.967] pred=원하다(0.307) target_conf=0.307 rank=1 hand=22
- strongest top1 predictions:
  - orig / trim_tail_0.4 pred=금욕(0.902) target_conf=0.013
  - zoom1.10_hflip / trim_both_0.2 pred=불신(0.822) target_conf=0.000
  - orig / trim_both_0.4 pred=금욕(0.810) target_conf=0.002

## merge2

### 1. 다리 expected=다리
- baseline: pred=다리 pred_conf=1.000 target_conf=1.000 target_rank=1 hand_frames=42
- best by expected target confidence:
  - orig / trim_tail_0.4 [0.0, 3.07] pred=다리(1.000) target_conf=1.000 rank=1 hand=37
  - zoom1.10 / trim_tail_0.4 [0.0, 3.07] pred=다리(1.000) target_conf=1.000 rank=1 hand=37
  - zoom1.20 / trim_tail_0.4 [0.0, 3.07] pred=다리(1.000) target_conf=1.000 rank=1 hand=37
  - zoom1.20 / trim_head_0.2 [0.2, 3.47] pred=다리(1.000) target_conf=1.000 rank=1 hand=36
  - zoom1.20 / trim_both_0.2 [0.2, 3.27] pred=다리(1.000) target_conf=1.000 rank=1 hand=36
- strongest top1 predictions:
  - orig / trim_tail_0.4 pred=다리(1.000) target_conf=1.000
  - zoom1.10 / trim_tail_0.4 pred=다리(1.000) target_conf=1.000
  - zoom1.20 / trim_tail_0.4 pred=다리(1.000) target_conf=1.000

### 2. 골절 expected=골절
- baseline: pred=골절 pred_conf=0.885 target_conf=0.885 target_rank=1 hand_frames=53
- best by expected target confidence:
  - zoom1.10 / extend_head_0.2 [3.27, 7.97] pred=골절(0.977) target_conf=0.977 rank=1 hand=56
  - zoom1.20 / base [3.47, 7.97] pred=골절(0.976) target_conf=0.976 rank=1 hand=48
  - zoom1.20 / trim_head_0.2 [3.67, 7.97] pred=골절(0.976) target_conf=0.976 rank=1 hand=48
  - zoom1.20 / extend_head_0.2 [3.27, 7.97] pred=골절(0.976) target_conf=0.976 rank=1 hand=48
  - zoom1.20 / extend_tail_0.2 [3.47, 8.17] pred=골절(0.974) target_conf=0.974 rank=1 hand=51
- strongest top1 predictions:
  - zoom1.10 / extend_head_0.2 pred=골절(0.977) target_conf=0.977
  - zoom1.20 / base pred=골절(0.976) target_conf=0.976
  - zoom1.20 / trim_head_0.2 pred=골절(0.976) target_conf=0.976

### 3. 아프다 expected=아프다
- baseline: pred=여자 pred_conf=0.742 target_conf=0.001 target_rank=40 hand_frames=29
- best by expected target confidence:
  - zoom1.20 / trim_head_0.2 [8.17, 10.433] pred=가다(0.217) target_conf=0.043 rank=4 hand=16
  - zoom1.20 / trim_both_0.2 [8.17, 10.257] pred=가다(0.217) target_conf=0.043 rank=4 hand=16
  - zoom1.20 / shift_right_0.2 [8.17, 10.433] pred=가다(0.217) target_conf=0.043 rank=4 hand=16
  - zoom1.20 / trim_head_0.4 [8.37, 10.433] pred=결점(0.162) target_conf=0.023 rank=7 hand=15
  - zoom1.20 / trim_both_0.4 [8.37, 10.057] pred=결점(0.162) target_conf=0.023 rank=7 hand=15
- strongest top1 predictions:
  - zoom1.10 / base pred=여자(0.812) target_conf=0.001
  - zoom1.10 / extend_tail_0.2 pred=여자(0.812) target_conf=0.001
  - zoom1.10 / extend_head_0.2 pred=여자(0.807) target_conf=0.001

## merge3

### 1. 소화불량 expected=소화불량
- baseline: pred=소화불량 pred_conf=0.994 target_conf=0.994 target_rank=1 hand_frames=54
- best by expected target confidence:
  - zoom1.20 / base [0.0, 4.5] pred=소화불량(0.996) target_conf=0.996 rank=1 hand=54
  - zoom1.20 / extend_head_0.2 [0.0, 4.5] pred=소화불량(0.996) target_conf=0.996 rank=1 hand=54
  - orig / base [0.0, 4.5] pred=소화불량(0.994) target_conf=0.994 rank=1 hand=54
  - orig / extend_head_0.2 [0.0, 4.5] pred=소화불량(0.994) target_conf=0.994 rank=1 hand=54
  - zoom1.20 / trim_head_0.2 [0.2, 4.5] pred=소화불량(0.994) target_conf=0.994 rank=1 hand=51
- strongest top1 predictions:
  - zoom1.20 / base pred=소화불량(0.996) target_conf=0.996
  - zoom1.20 / extend_head_0.2 pred=소화불량(0.996) target_conf=0.996
  - orig / base pred=소화불량(0.994) target_conf=0.994

### 2. 어떻게 expected=어떻게
- baseline: pred=적중하다 pred_conf=0.146 target_conf=0.022 target_rank=9 hand_frames=32
- best by expected target confidence:
  - zoom1.10 / extend_tail_0.2 [4.5, 7.433] pred=뽀뽀(0.211) target_conf=0.194 rank=2 hand=36
  - zoom1.10 / trim_tail_0.2 [4.5, 7.033] pred=뽀뽀(0.229) target_conf=0.094 rank=3 hand=31
  - zoom1.10 / trim_tail_0.4 [4.5, 6.833] pred=뽀뽀(0.226) target_conf=0.093 rank=4 hand=28
  - zoom1.10 / base [4.5, 7.233] pred=뽀뽀(0.210) target_conf=0.088 rank=3 hand=33
  - zoom1.10 / extend_head_0.2 [4.3, 7.233] pred=곰방대(0.386) target_conf=0.076 rank=4 hand=35
- strongest top1 predictions:
  - zoom1.20_hflip / trim_both_0.2 pred=뉴질랜드(0.793) target_conf=0.000
  - zoom1.20_hflip / trim_head_0.2 pred=뉴질랜드(0.789) target_conf=0.000
  - zoom1.20 / trim_both_0.2 pred=낙인(0.742) target_conf=0.001

### 3. 치료 expected=치료
- baseline: pred=면역 pred_conf=0.117 target_conf=0.021 target_rank=7 hand_frames=46
- best by expected target confidence:
  - zoom1.20 / shift_left_0.2 [7.033, 11.065] pred=면역(0.256) target_conf=0.170 rank=3 hand=45
  - zoom1.20 / trim_head_0.2 [7.433, 11.233] pred=면역(0.349) target_conf=0.156 rank=2 hand=42
  - zoom1.20 / shift_right_0.2 [7.433, 11.233] pred=면역(0.349) target_conf=0.156 rank=2 hand=42
  - zoom1.20 / trim_head_0.4 [7.633, 11.233] pred=면역(0.450) target_conf=0.134 rank=3 hand=40
  - zoom1.20 / trim_both_0.4 [7.633, 10.865] pred=면역(0.406) target_conf=0.133 rank=2 hand=36
- strongest top1 predictions:
  - zoom1.20_hflip / trim_tail_0.4 pred=매형(0.846) target_conf=0.000
  - zoom1.20_hflip / trim_tail_0.2 pred=매형(0.843) target_conf=0.000
  - zoom1.20_hflip / base pred=매형(0.835) target_conf=0.000

