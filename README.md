### 2022 KT-ETRI 이상치 탐지 대회 three0_s 제출      

#### IP

- Requisites   

```
anaconda 가상환경 설정 + (cuda+cudnn+pytorch 설정 py3.8_cuda11.3_cudnn8.2.0_0)
pip install -r requirements.txt
```

1. IP_data_process.ipynb 실행 -> 학습 데이터 준비   
2. train & predict
```
python train.py
python IP.py
```
3. predict
```
python IP.py
```

#### Media
1. media.ipynb 열기
2. 위에서부터 차례대로 데이터 준비, 학습 및 추론을 실행한다.