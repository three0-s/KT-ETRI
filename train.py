import random
import numpy as np
import torch.optim as optim
from model import Attention, Encoder, Decoder, Seq2Seq, init_weights, count_parameters, train_an_epoch, \
                evaluate, epoch_time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import joblib
import time
from torch.utils.tensorboard import SummaryWriter

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

train_npy = joblib.load(f"data/{TASK.upper()}/{TASK.lower()}_train_ts_{N_TIME_STEP}_{INPUT_DIM}f.pkl")
test_npy = joblib.load(f"data/{TASK.upper()}/{TASK.lower()}_valid_ts_{N_TIME_STEP}_{INPUT_DIM}f.pkl")

train_data = TensorDataset(torch.from_numpy(train_npy))
test_data = TensorDataset(torch.from_numpy(test_npy))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(ENC_N_LAYERS, INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(attn, DEC_N_LAYERS, OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)
model.apply(init_weights)

print(f'The model has {count_parameters(model):,} trainable parameters')
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.005)
loss = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, pct_start=0.2, div_factor=50,
                                                steps_per_epoch=len(train_loader), epochs=N_EPOCHS,anneal_strategy='cos')


best_valid_loss = float('inf')
writer = SummaryWriter()
early_stop_count = 0

for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss = train_an_epoch(model, train_loader, optimizer, loss, CLIP, device, scheduler, tf_ratio=0.75)
    test_loss = evaluate(model, test_loader, loss, device)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    writer.add_scalar('Train Loss', train_loss, epoch)
    writer.add_scalar('Validation Loss', test_loss, epoch)
    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
    
    if test_loss < best_valid_loss:
        best_valid_loss = test_loss
        torch.save(model.state_dict(), CHECKPOINT_DIR+NAME+'.pth')
        early_stop_count = 0
    else:
        # early stopping (endure at most 12 epochs of not getting better scores in validation set)
        if early_stop_count >= 12:
            print('='*50)
            print(f'\nEarly stopping at {epoch+1} epoch..')
            print(f'Best validation MSE :{best_valid_loss}\n')
            print('='*50)
            break
        early_stop_count+=1

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\tValidation Loss: {test_loss:.3f}')
