# Seq2Seq model with attention
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from copy import copy

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_an_epoch(model, iterator, optimizer, criterion, clip, device, scheduler, tf_ratio=1.):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch[0].float().to(device)
        trg = copy(batch[0]).float().to(device)
        
        optimizer.zero_grad()
        output = model(src, trg, tf_ratio).permute(1, 0, 2)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0].float().to(device)
            trg = copy(batch[0]).float().to(device)
            output = model(src, trg, 0).permute(1, 0, 2) #turn off teacher forcing
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class Encoder(nn.Module):
    def __init__(self, num_layers=2, input_dim=10, emb_dim=64, enc_hid_dim=128, dec_hid_dim=128, dropout=0.5):
        super().__init__()
        self.embedding = nn.Linear(input_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True, batch_first = True, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        embedded = self.norm(embedded)
        #embedded = [batch size, n_time_steps, emb dim]
        outputs, hidden = self.rnn(embedded)
        #outputs = [batch size, n_time_steps, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        #hidden = [batch size, dec hid dim]
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.norm = nn.LayerNorm((enc_hid_dim * 2) + dec_hid_dim)
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        src_len = encoder_outputs.shape[1]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        energy = torch.tanh(self.attn(self.norm(torch.cat((hidden, encoder_outputs), dim = 2)))) 
        #energy = [batch size, src len, dec hid dim]
        attention = self.v(energy).squeeze(2)
        #attention= [batch size, src len]
        return F.softmax(attention, dim=1)



class Decoder(nn.Module):
    def __init__(self, attention, num_layer=1, output_dim=10, emb_dim=64, enc_hid_dim=128, dec_hid_dim=128, dropout=0.5):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Linear(output_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, num_layers=num_layer)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, dec_hid=None):
        #input = [batch size, n_features]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [batch size, n_time_steps, enc hid dim * 2]
        input = input.unsqueeze(1)
        #input = [batch size, 1, n_features]
        embedded = self.dropout(self.embedding(input)).permute(1, 0, 2)
        embedded = self.norm(embedded)
        #embedded = [1, batch_size, emb dim]
        a = self.attention(hidden, encoder_outputs)   
        #a = [batch size, src len]
        a = a.unsqueeze(1)
        #a = [batch size, 1, src len]
        
        weighted = torch.bmm(a, encoder_outputs)
        #weighted = [batch size, 1, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)
        #weighted = [1, batch size, enc hid dim * 2]
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
        if dec_hid == None:
            output, dec_hid = self.rnn(rnn_input)
        else:
            output, dec_hid = self.rnn(rnn_input, dec_hid)
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        #prediction = [batch size, output dim]
        return prediction, dec_hid


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
                
        #first input to the decoder (start token = [0, ..., 0])
        input = torch.zeros_like(trg[:, 0, :]).to(self.device)
        dec_hid = None
        for t in range(0, trg_len-1):
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, dec_hid = self.decoder(input, hidden, encoder_outputs, dec_hid)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            hidden = dec_hid[-1, ...]
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[:, t, :] if teacher_force else output

        return outputs