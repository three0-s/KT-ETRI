# ======================================================================
# 분야 #1: IP 네트워크
#
# 실시간 IP 할당 개수 추이를 기반으로 이상 발생 시점을 탐지하는 문제입니다.
# DHCP란 Dynamic Host Configuration Protocol의 약자로, 클라이언트(단말)의 요청에 따라 IP 주소를 동적으로 할당 및 관리합니다.
# DHCP 서버는 서버와 클라이언트를 중개하는 방식으로 요청 단말에게 IP를 할당합니다
# IP 네트워크 문제에서는 DHCP 장비 1종으로부터 수집된 10분 주기의 IP 세션 데이터 12개월치가 제공됩니다.
# 주어진 데이터를 활용하여, 2021년 하반기(7월 1일 - 12월 31일) IP 할당 프로세스의 이상 발생 시점을 예측하세요.
#
# 데이터 파일명: DHCP.csv
#
# 데이터 컬럼 설명:
# Timestamp: [YYYYMMDD_HHmm(a)-HHmm(b)] 형식을 가지며, 수집 범위는 YYYY년 MM월 DD일 HH시 mm분(a) 부터 HH시 mm분(b)입니다.
# Svr_detect: DHCP 프로세스에서 단위 시간 내 클라이언트인 단말들이 DHCP 서버에게 연결을 요청한 횟수입니다.
# Svr_connect: DHCP 프로세스에서 단위 시간 내 클라이언트인 단말들에게 DHCP 서버와 연결이 확립됨을 나타내는 횟수입니다.
# Ss_request: DHCP 프로세스에서 단위 시간 내 서버에 연결된 단말들이 IP 할당을 요청한 횟수입니다.
# Ss_established: IP 할당 요청을 받은 DHCP 서버가 클라이언트에게 IP가 할당됨을 나타내는 횟수입니다.
#
# * 데이터에는 일부 결측치가 존재합니다
# ======================================================================
import random
import numpy as np
from model import Attention, Encoder, Decoder, Seq2Seq
import torch
from torch.utils.data import TensorDataset, DataLoader
import joblib
from tqdm import tqdm
import pandas as pd

# ================================================

BATCH_SIZE = 256
SEED = 42
ENC_EMB_DIM = DEC_EMB_DIM = 256   # N
ENC_HID_DIM = DEC_HID_DIM = ENC_EMB_DIM * 2  # 2N
INPUT_DIM = OUTPUT_DIM = 190
ENC_N_LAYERS = DEC_N_LAYERS = 4 # L
ENC_DROPOUT = DEC_DROPOUT = 0.5
N_EPOCHS = 100
CLIP = 1
ANRATIO = 0.0085
N_TIME_STEP = 12
TASK = 'IP'
CHECKPOINT_DIR = 'models/'

NAME = f'{TASK}_seq2seq_attn_{INPUT_DIM}f_{N_TIME_STEP}TS_tf075_{ENC_EMB_DIM}N_{ENC_N_LAYERS}L_{N_EPOCHS}e'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================================
def save_pred(model, test_loader, ratio=0.005):
    # TODO: 모델을 활용해, 2021년 하반기 전체에 대한 예측을 수행하세요!
    model.eval()
    loss_list = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            origin = batch[0].float().to(device)
            encoded = model(origin, origin, 0).permute(1, 0, 2) 
            encoded = encoded[:, -1, :] # We only need the last time step
            
            origin = origin[:, -1, :]
            loss = torch.nn.MSELoss(reduction='none')(origin, encoded)
            loss_list.append(loss.detach().cpu().numpy())
    
    losses = np.mean(np.concatenate(loss_list, axis=0), axis=1)

    pred = np.zeros_like(losses)
    thr = np.percentile(losses, [(1-ratio)*100])[0]
    print(f'Threshold: {thr}')
    # abnormal if loss is smaller than the threshold
    pred[losses > thr] = 1
    # 예측된 결과를 제출하기 위한 포맷팅
    answer = pd.DataFrame(pred, columns=['Prediction'])
    num = np.sum(answer['Prediction']==1)
    print(f'Identified err: {num}') 
    print(f'예측 결과. \n{answer}\n')  # TODO: 제출 전 row size "26496" 확인
    answer.to_csv(f'{TASK}_answer_{NAME}_{ratio}.csv', index=False)  # 제출용 정답지 저장

# TODO: 제출 파일은 2021년 7월 1일 00시 00분-10분 부터 2021년 12월 31일 23시 50분-00분 구간의 이상 이벤트를 예측한
#  .csv 형식으로 저장해야 합니다.
#  예측 데이터프레임의 크기는 [26496 * 1]입니다.

if __name__ == '__main__':
    test_npy = joblib.load(f"data/{TASK.upper()}/{TASK.lower()}_test_ts_{N_TIME_STEP}_{INPUT_DIM}f.pkl")
    test_data = TensorDataset(torch.from_numpy(test_npy))
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(ENC_N_LAYERS, INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(attn, DEC_N_LAYERS, OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_DIR+NAME+'.pth', map_location=device))
    print(f"Model loaded: \n{model}\n")
    print("="*60)
    # 예측 결과 저장
    save_pred(model, test_loader, ratio=ANRATIO)
