
import time
import re
import os
import math
import collections
#import evaluation
import tokenization
import numpy as np
import random
from random import choice
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from torchvision import transforms, models
from SGD import SGD_GC


train_text_file = '/media/lab/Backup Plus/CGMVQA-master-Change/ImageClef-2019-VQA-Med-Training/QAPairsByCategory/C1_Modality_train.txt'
#valid_text_file = './ImageClef-2019-VQA-Med-Validation/All_QA_Pairs_val.txt'
test_text_file = '/media/lab/Backup Plus/CGMVQA-master-Change/VQAMed2019Test/2019_gt_mod_all.txt'
train_image_file = '/media/lab/Backup Plus/CGMVQA-master-Change/ImageClef-2019-VQA-Med-Training/Train_images/'
#valid_image_file = './ImageClef-2019-VQA-Med-Validation/Val_images/'
test_image_file = '/media/lab/Backup Plus/CGMVQA-master-Change/VQAMed2019Test/VQAMed2019_Test_Images/'
vocab_file = './bio_vocab.txt'
save_dir = './save_mod/'
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))
token_id = tokenizer.convert_tokens_to_ids
vocab_size = 30000
max_len = 20
gen_len = 33
heads = 12
epochs = 40
lr = 0.0001
clip = True
dim = 312
drop = 0.0
n_layers = 4
batch_size = 64

# model
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Transfer(nn.Module):
    def __init__(self, model=models.resnet152(pretrained=True), trans=transforms.Compose([transforms.Resize((224,224)), transforms.RandomResizedCrop(224,scale=(0.95,1.05),ratio=(0.95,1.05)),transforms.RandomRotation(5),transforms.ColorJitter(brightness=0.05,contrast=0.05,saturation=0.05,hue=0.05),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),img_to_gray = transforms.Compose([transforms.Resize((224,224)),transforms.Grayscale(1),transforms.ToTensor()])):
        super(Transfer,self).__init__()
        self.model = model
        self.copy = 0
        self.lstm = nn.LSTM(input_size=224, hidden_size=312,num_layers=2,batch_first=True,bidirectional=False)
        self.gru = nn.GRU(input_size=224, hidden_size=312,num_layers=2,batch_first=True,bidirectional=False)
        
        for p in self.parameters():
            p.requires_grad=False
        self.trans = trans
        self.img_to_gray = img_to_gray
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(2048, dim, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap2 = nn.AdaptiveAvgPool2d((1,1))
        self.conv3 = nn.Conv2d(1024, dim, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap3 = nn.AdaptiveAvgPool2d((1,1))
        self.conv4 = nn.Conv2d(512, dim, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap4 = nn.AdaptiveAvgPool2d((1,1)) 
        self.conv5 = nn.Conv2d(256, dim, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap5 = nn.AdaptiveAvgPool2d((1,1))
        self.conv7 = nn.Conv2d(64, dim, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.gap7 = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, name):
        v_list = []
        v_gray_list = []                                                                                                                                                                                                                                                                                                                                                                                                         
        for v in name:
            if os.path.isfile(train_image_file + v + '.jpg') == True:
                 f = Image.open(train_image_file + v + '.jpg')
            else:
                f = Image.open(test_image_file + v + '.jpg')

            imag = self.trans(f)
            imag_gray = self.img_to_gray(f)
            if len(imag) == 1:
                imag = torch.cat([imag,imag,imag],dim=0)
            v_list.append(imag)
            f.close()
            v_gray_list.append(imag_gray)
            f.close()
        v_list = torch.stack(v_list).to(device)
        v_gray_list = torch.stack(v_gray_list).to(device)
        v_gru = np.squeeze(v_gray_list)
        if(v_gru.ndim == 3):
            l2,_ = self.gru(v_gru,None)
            l2 = l2[:,-1,:]
            l2 = np.squeeze(l2)
        else:
            v_gru = torch.reshape(v_gru,(1,224,224))
            l2,_ = self.gru(v_gru,None)
            l2 = l2[:,-1,:]
            l2 = np.squeeze(l2)
            l2 = l2.expand(64,312)
        
        modules2 = list(self.model.children())[:-2]
        fix2 = nn.Sequential(*modules2)
        v_2 = self.gap2(self.relu(self.conv2(fix2(v_list)))).view(-1,dim)
        v_2 = torch.cat([v_2,l2],dim=0).to(device)   
        modules3 = list(self.model.children())[:-3]
        fix3 = nn.Sequential(*modules3)
        v_3 = self.gap3(self.relu(self.conv3(fix3(v_list)))).view(-1,dim)
        v_3 = torch.cat([v_3,l2],dim=0).to(device)
        modules4 = list(self.model.children())[:-4]
        fix4 = nn.Sequential(*modules4)
        v_4 = self.gap4(self.relu(self.conv4(fix4(v_list)))).view(-1,dim)
        v_4 = torch.cat([v_4,l2],dim=0).to(device)
        modules5 = list(self.model.children())[:-5]
        fix5 = nn.Sequential(*modules5)
        v_5 = self.gap5(self.relu(self.conv5(fix5(v_list)))).view(-1,dim)
        v_5 = torch.cat([v_5,l2],dim=0).to(device)
        modules7 = list(self.model.children())[:-7]
        fix7 = nn.Sequential(*modules7)
        v_7 = self.gap7(self.relu(self.conv7(fix7(v_list)))).view(-1,dim)
        v_7 = torch.cat([v_7,l2],dim=0).to(device)
        return v_2, v_3, v_4, v_5, v_7
    
class Embeddings(nn.Module):
    def __init__(self,max_len):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.word_embeddings_2 = nn.Linear(128, dim, bias=False)
        self.position_embeddings = nn.Embedding(max_len, dim)
        self.type_embeddings = nn.Embedding(3, dim)
        self.LayerNorm = nn.LayerNorm(dim, eps=1e-12)
        self.dropout = nn.Dropout(drop)
        self.len = max_len
    def forward(self, input_ids, segment_ids, position_ids=None):
        input_ids = torch.cuda.LongTensor(input_ids)
        segment_ids = torch.cuda.LongTensor(segment_ids)
        if position_ids is None:
            position_ids = torch.arange(self.len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        words_embeddings = self.word_embeddings_2(words_embeddings)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.type_embeddings(segment_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)
        self.scores = None
        self.n_heads = heads
    def forward(self, x, mask):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (self.split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = self.merge_last(h, 2)
        self.scores = scores
        return h
    def split_last(self, x, shape):
        shape = list(shape)
        assert shape.count(-1) <= 1
        if -1 in shape:
            shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
        return x.view(*x.size()[:-1], *shape)
    def merge_last(self, x, n_dims):
        s = x.size()
        assert n_dims > 1 and n_dims < len(s)
        return x.view(*s[:-n_dims], -1)

class PositionWiseFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim*4)
        self.fc2 = nn.Linear(dim*4, dim)
    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))

class BertLayer(nn.Module):
    def __init__(self, share='all', norm='pre'):
        super(BertLayer, self).__init__()
        self.share = share
        self.norm_pos = norm
        self.norm1 = nn.LayerNorm(dim, eps=1e-12)
        self.norm2 = nn.LayerNorm(dim, eps=1e-12)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
        if self.share == 'ffn':
            self.attention = nn.ModuleList([MultiHeadedSelfAttention() for _ in range(n_layers)])
            self.proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
            self.feedforward = PositionWiseFeedForward()
        elif self.share == 'att':
            self.attention = MultiHeadedSelfAttention()
            self.proj = nn.Linear(dim, dim)
            self.feedforward = nn.ModuleList([PositionWiseFeedForward() for _ in range(n_layers)])
        elif self.share == 'all':
            self.attention = MultiHeadedSelfAttention()
            self.proj = nn.Linear(dim, dim)
            self.feedforward = PositionWiseFeedForward()
        elif self.share == 'none':
            self.attention = nn.ModuleList([MultiHeadedSelfAttention() for _ in range(n_layers)])
            self.proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
            self.feedforward = nn.ModuleList([PositionWiseFeedForward() for _ in range(n_layers)])
    def forward(self, hidden_states, attention_mask, layer_num):
        attention_mask = torch.cuda.LongTensor(attention_mask)
        if self.norm_pos == 'pre':
            if isinstance(self.attention, nn.ModuleList):
                h = self.proj[layer_num](self.attention[layer_num](self.norm1(hidden_states), attention_mask))
            else:
                h = self.proj(self.attention(self.norm1(hidden_states), attention_mask))
            out = hidden_states + self.drop1(h)
            if isinstance(self.feedforward, nn.ModuleList):
                h = self.feedforward[layer_num](self.norm1(out))
            else:
                h = self.feedforward(self.norm1(out))
            out = out + self.drop2(h)
        if self.norm_pos == 'post':
            if isinstance(self.attention, nn.ModuleList):
                h = self.proj[layer_num](self.attention[layer_num](hidden_states, attention_mask))
            else:
                h = self.proj(self.attention(hidden_states, attention_mask))
            out = self.norm1(hidden_states + self.drop1(h))
            if isinstance(self.feedforward, nn.ModuleList):
                h = self.feedforward[layer_num](out)
            else:
                h = self.feedforward(out)
            out = self.norm2(out + self.drop2(h))
        return out

class Transformer(nn.Module):
    def __init__(self,max_len=max_len):
        super().__init__()
        self.embed = Embeddings(max_len)
        self.trans = Transfer()
        self.blocks = BertLayer(share='none', norm='pre')
        self.n_layers = n_layers
    def forward(self, name, x, seg, mask):
        v_2, v_3, v_4, v_5, v_7 = self.trans(name)
        h = self.embed(x, seg)
        for i in range(len(h)):
            h[i][1] = v_2[i]
        for i in range(len(h)):
            h[i][2] = v_3[i]
        for i in range(len(h)):
            h[i][3] = v_4[i]
        for i in range(len(h)):
            h[i][4] = v_5[i]
        for i in range(len(h)):
            h[i][5] = v_7[i]
        for i in range(self.n_layers):
            h = self.blocks(h, mask, i)
        return h

class mod_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()
        self.fc1 = nn.Linear(dim, dim)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(dim, 43)
    def forward(self, name, input_ids, segment_ids, input_mask):
        h = self.transformer(name, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h[:, 0]))
        logits_clsf = self.classifier(pooled_h)
        return logits_clsf



class mod_yn_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()
        self.fc1 = nn.Linear(dim, dim)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(dim, 2)
    def forward(self, name, input_ids, segment_ids, input_mask):
        h = self.transformer(name, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h[:, 0]))
        logits_clsf = self.classifier(pooled_h)
        return logits_clsf

# data
def ques_standard(text):
    temp = text.strip('?').split(' ')
    temp_list = []
    for i in range(len(temp)):
        if temp[i] != '':
            temp[i] = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+—�?)?【】“”！，。？、~@#�?…�?*（）]+ ", "", temp[i].lower())
            temp_list.append(temp[i].replace('-',' '))
    return ' '.join(temp_list)

def extract_data(file, start, end):
    imag, ques, answ = [],[],[]
    f = open(file,'r')
    lines = f.readlines()[start:end]
    for line in lines:
        imag.append(line.split('|')[0])
        ques.append(ques_standard(line.split('|')[-2]))
        answ.append(line.strip().split('|')[-1])
    f.close()
    return imag, ques, answ

def clsf_data(start,end,mode='class'):
    _imag, _ques, _answ = extract_data(train_text_file, start, end)
#    _imag2, _ques2, _answ2 = extract_data(valid_text_file, start2, end2)
#    _imag, _ques, _answ = _imag1, _ques1, _answ1
    yn_imag, yn_ques, yn_answ = [], [], []
    ge_imag, ge_ques, ge_answ = [], [], []
    for i in range(len(_answ)):
        if _answ[i] in ['yes','no','Yes','No']:
            yn_imag.append(_imag[i])
            yn_ques.append(_ques[i])
            yn_answ.append(_answ[i])
        else:
            ge_imag.append(_imag[i])
            ge_ques.append(_ques[i])
            ge_answ.append(_answ[i])
    yn_list = list(set(yn_answ))
    yn_dict = {yn_list[i]:i for i in range(len(yn_list))}
    for i in range(len(yn_answ)):
        yn_answ[i] = yn_dict[yn_answ[i]]
    ge_list = list(set(ge_answ))
    ge_dict = {ge_list[i]:i for i in range(len(ge_list))}
    for i in range(len(ge_answ)):
        ge_answ[i] = ge_dict[ge_answ[i]]
    if mode == 'yn':
        return yn_imag, yn_ques, yn_answ, yn_dict
    else:
        return ge_imag, ge_ques, ge_answ, ge_dict

def balance_data(start,end,mode='class'):
    _imag, _ques, _answ, _dict = clsf_data(start,end,mode)
    freq = collections.Counter(_answ)
    sup_imag, sup_ques, sup_answ = [], [], []
    for key in freq.keys():
        if freq[key] > 3:
            pad = freq[max(freq)] - freq[key]
            temp_answ = [k for k, x in enumerate(_answ) if x == key]
            for i in range(pad):
                sup_answ.append(key)
                j = choice(temp_answ)
                sup_imag.append(_imag[j])
                sup_ques.append(_ques[j])
    imag = _imag + sup_imag
    ques = _ques + sup_ques
    answ = _answ + sup_answ
    state = np.random.get_state()
    np.random.shuffle(imag)
    np.random.set_state(state)
    np.random.shuffle(ques)
    np.random.set_state(state)
    np.random.shuffle(answ)
    return imag, ques, answ, _dict

mod_imag, mod_ques, mod_answ, mod_dict = balance_data(0,2337)
mod_yn_imag, mod_yn_ques, mod_yn_answ, mod_yn_dict = balance_data(2338,3700,mode='yn')

def gene_data(start,end):
    _imag, _ques, _answ = extract_data(train_text_file, start, end)
#    _imag2, _ques2, _answ2 = extract_data(valid_text_file, start2, end2)
#    _imag, _ques, _answ = _imag1+_imag2, _ques1+_ques2, _answ1+_answ2
    ge_imag, ge_ques, ge_answ = [], [], []
    for i in range(len(_answ)):
        if _answ[i] not in ['yes','no']:
            ge_imag.append(_imag[i])
            ge_ques.append(_ques[i])
            ge_answ.append(_answ[i])
    state = np.random.get_state()
    np.random.shuffle(ge_imag)
    np.random.set_state(state)
    np.random.shuffle(ge_ques)
    np.random.set_state(state)
    np.random.shuffle(ge_answ)
    return ge_imag, ge_ques, ge_answ, ge_imag, ge_ques, ge_answ

#abn_train_imag, abn_train_ques, abn_train_answ, abn_valid_imag, abn_valid_ques, abn_valid_answ = gene_data(4,4,6,6)

def data_loader(imag, ques, answ, batch_size, max_len):
    state = np.random.get_state()
    np.random.shuffle(imag)
    np.random.set_state(state)
    np.random.shuffle(ques)
    np.random.set_state(state)
    np.random.shuffle(answ)
    count = 0
    while count < len(imag):
        batch = []
        if batch_size < len(imag) - count:
            size = batch_size
        else: 
            size = len(imag) - count
        for _ in range(size):
            classes = answ[count]
            imag_name = imag[count]
            part1 = [0 for _ in range(5)]
            part2 = token_id(tokenize(ques[count]))
            tokens = token_id(['[CLS]']) + part1 + token_id(['[SEP]']) + part2[:max_len-8] + token_id(['[SEP]'])
            segment_ids = [0]*(len(part1)+2) + [1]*(len(part2[:max_len-8])+1)
            input_mask = [1]*len(tokens)
            n_pad = max_len - len(tokens)
            tokens.extend([0]*n_pad)
            segment_ids.extend([0]*n_pad)
            input_mask.extend([0]*n_pad)
            batch.append((imag_name, tokens, segment_ids, input_mask, classes))
            count += 1
        yield batch

def truncate_tokens(tokens_a, tokens_b, gen_len):
    while True:
        if len(tokens_a) + len(tokens_b) <= gen_len:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def gene_loader(imag, ques, answ, batch_size, gen_len):
    count = 0
    while count < len(imag):
        batch = []
        if batch_size < len(imag) - count:
            size = batch_size
        else: 
            size = len(imag) - count
        for _ in range(size):
            imag_name = imag[count]
            part1 = [0 for _ in range(5)]
            part2 = token_id(tokenize(ques[count]))
            part3 = token_id(tokenize(answ[count]))
            truncate_tokens(part2, part3, gen_len-9)
            tokens = token_id(['[CLS]']) + part1 + token_id(['[SEP]']) + part2 + token_id(['[SEP]']) + part3 + token_id(['[SEP]'])
            masked_tokens, masked_pos = [], []
            num = random.randint(len(part1+part2)+3,len(part1+part2+part3)+3)
            masked_pos.append(num)
            masked_tokens.append(tokens[num])
            masked_weights = [1]*len(masked_tokens)
            tokens[num] = token_id(['[MASK]'])[0]
            for i in range(num+1,len(tokens)):
                tokens[i] = 0
            segment_ids = [0]*(len(part1)+2) + [1]*(len(part2)+1) + [2]*(num-len(part1+part2)-2)
            input_mask = [1]*(num+1)
            n_pad = gen_len - num - 1
            tokens.extend([0]*(gen_len - len(tokens)))
            segment_ids.extend([0]*n_pad)
            input_mask.extend([0]*n_pad)
            batch.append((imag_name, tokens, segment_ids, input_mask, masked_tokens, masked_pos, masked_weights))
            count += 1
        yield batch

def get_loss(model, batch):
    imag_name, tokens, segment_ids, input_mask, classes = zip(*batch)
    logits_clsf = model(imag_name, tokens, segment_ids, input_mask)
    loss_clsf = nn.CrossEntropyLoss()(logits_clsf, torch.cuda.LongTensor(classes))
    return loss_clsf

def gene_loss(model, batch):
    imag_name, tokens, segment_ids, input_mask, masked_tokens, masked_pos, masked_weights = zip(*batch)
    logits_lm = model(imag_name, tokens, segment_ids, input_mask, masked_pos)
    masked_tokens, masked_weights = torch.cuda.LongTensor(masked_tokens), torch.cuda.LongTensor(masked_weights)
    loss_lm = nn.CrossEntropyLoss(reduction='none')(logits_lm.transpose(1, 2), masked_tokens)
    loss_lm = (loss_lm*masked_weights.float()).mean()
    return loss_lm

def train(model, iterator, optimizer, epoch, mode='class'):
    model.train()
    epoch_loss, count = 0, 0
    iter_bar = tqdm(iterator, desc='Training')
    for _, batch in enumerate(iter_bar):
        count += 1
        optimizer.zero_grad()
        if mode == 'class':
            loss = get_loss(model, batch)
        else:
            loss = gene_loss(model, batch)
        loss = loss.mean()
        loss.backward()
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  #梯度裁剪
        optimizer.step()
        iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())
        epoch_loss += loss.item()
    return epoch_loss / count

def valid(model, iterator, epoch, mode='class'):
    model.eval()
    epoch_loss, count = 0, 0
    with torch.no_grad():
        iter_bar = tqdm(iterator, desc='Validation')
        for _, batch in enumerate(iter_bar):
            count += 1
            if mode == 'class':
                loss = get_loss(model, batch)
            else:
                loss = gene_loss(model, batch)
            loss = loss.mean()
            iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())
            epoch_loss += loss.item()
    return epoch_loss / count

def test_data(test_imag, test_ques, i):
    imag_name = test_imag[i]
    part1 = [0 for _ in range(5)]
    part2 = token_id(tokenize(test_ques[i]))
    tokens = token_id(['[CLS]']) + part1 + token_id(['[SEP]']) + part2[:max_len-8] + token_id(['[SEP]'])
    segment_ids = [0]*(len(part1)+2) + [1]*(len(part2[:max_len-8])+1)
    input_mask = [1]*len(tokens)
    n_pad = max_len - len(tokens)
    tokens.extend([0]*n_pad)
    segment_ids.extend([0]*n_pad)
    input_mask.extend([0]*n_pad)
    return (test_ques[i], imag_name, tokens, segment_ids, input_mask)

def get_answ(model, dict_op, imag_name, tokens, segment_ids, input_mask):
    model.eval()
    with torch.no_grad():
        pred = model([imag_name], [tokens], [segment_ids], [input_mask])
    out = dict_op[int(np.argsort(pred.cpu())[:,-1:][0][0])]
    return out

def train_model(epochs,model,imag_t,ques_t,answ_t,imag_v,ques_v,answ_v,log_name,threshold,start,end,yn_mode,dict_op):
    log_file = save_dir+log_name+'.txt'
    with open(log_file, 'w') as log_f:
        log_f.write('epoch, train_loss, valid_loss\n')
    optimizer = optim.SGD(model.parameters(),lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True)
    stop = 0
    loss_list = []
    result_dict = {}
    for epoch in range(epochs):
        train_iterator = data_loader(imag_t,ques_t,answ_t, batch_size, max_len)
        valid_iterator = data_loader(imag_v,ques_v,answ_v, batch_size, max_len)
        print('Epoch: ' + str(epoch+1))
        train_loss = train(model, train_iterator, optimizer, epoch)
        valid_loss = valid(model, valid_iterator, epoch)
        scheduler.step(valid_loss)
        loss_list.append(valid_loss)
        with open(log_file, 'a') as log_f:
            log_f.write('{epoch},{train_loss: 3.5f},{valid_loss: 3.5f}\n'.format(epoch=epoch+1, train_loss=train_loss, valid_loss=valid_loss))   
        if valid_loss == min(loss_list) and valid_loss < threshold:
            stop, count = 0, 0
            gt_temp, answ_temp = [], []
            for i in range(start, end):
                ques_name, imag_name, tokens, segment_ids, input_mask = test_data(test_imag, test_ques, i)

                if test_ques[i] in mod_yn_ques:
                    result_dict[str(i)] = get_answ(model, dict_op, imag_name, tokens, segment_ids, input_mask)
                    gt_temp.append(test_answ[i])
                    answ_temp.append(get_answ(model, dict_op, imag_name, tokens, segment_ids, input_mask))
                else:
                    if test_ques[i] not in mod_yn_ques:
                        result_dict[str(i)] = get_answ(model, dict_op, imag_name, tokens, segment_ids, input_mask)
                        gt_temp.append(test_answ[i])
                        answ_temp.append(get_answ(model, dict_op, imag_name, tokens, segment_ids, input_mask))
#                    
                        
            for i in range(len(gt_temp)):
                if gt_temp[i] == answ_temp[i]:
                    count += 1
            torch.save(model.state_dict(), os.path.join(save_dir, log_name+'_'+str(count/len(gt_temp))[:5]+'_'+str(valid_loss)[:5]+'.pt'))
        else:
            stop += 1
            if stop > 10:
                break
    return result_dict, min(loss_list)

def train_gene(model,imag_t,ques_t,answ_t,imag_v,ques_v,answ_v,log_name,threshold,start,end,dict_op,loss_mode='gene'):
    log_file = save_dir+log_name+'.txt'
    with open(log_file, 'w') as log_f:
        log_f.write('epoch, train_loss, valid_loss\n')
    optimizer = optim.Adam(model.parameters(),lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True)
    stop = 0
    loss_list = []
    result_dict = {}
    for epoch in range(epochs):
        train_iterator = gene_loader(imag_t,ques_t,answ_t, batch_size, gen_len)
        valid_iterator = gene_loader(imag_v,ques_v,answ_v, batch_size, gen_len)
        print('Epoch: ' + str(epoch+1))
        train_loss = train(model, train_iterator, optimizer, epoch, loss_mode)
        valid_loss = valid(model, valid_iterator, epoch, loss_mode)
        scheduler.step(valid_loss)
        loss_list.append(valid_loss)
        with open(log_file, 'a') as log_f:
            log_f.write('{epoch},{train_loss: 3.5f},{valid_loss: 3.5f}\n'.format(epoch=epoch+1, train_loss=train_loss, valid_loss=valid_loss))   
        if valid_loss == min(loss_list) and valid_loss < threshold:
            stop = 0
            for i in range(start, end):
                imag_name = test_imag[i]
                part1 = [0 for _ in range(5)]
                part2 = token_id(tokenize(test_ques[i]))
                tokens = token_id(['[CLS]']) + part1 + token_id(['[SEP]']) + part2 + token_id(['[SEP]']) + token_id(['[MASK]'])
                masked_pos = [len(part1+part2)+3]
                segment_ids = [0]*(len(part1)+2) + [1]*(len(part2)+1) + [2]*(1)
                input_mask = [1]*(len(part1+part2)+4)
                n_pad = gen_len - len(part1+part2) - 4
                tokens.extend([0]*(n_pad))
                segment_ids.extend([0]*n_pad)
                input_mask.extend([0]*n_pad)
                output = []
                for k in range(n_pad):
                    with torch.no_grad():
                        pred = model([imag_name], [tokens], [segment_ids], [input_mask], [masked_pos])
                        out = int(np.argsort((pred.cpu())[0][0])[-1])
                        output.append(abn_dict_op[out])
                        if out == 102:
                            break
                        else:
                            tokens[len(part1+part2)+3+k] = out
                            tokens[len(part1+part2)+4+k] = token_id(['[MASK]'])[0]
                            masked_pos = [len(part1+part2)+4+k]
                            segment_ids = [0]*(len(part1)+2) + [1]*(len(part2)+1) + [2]*(2+k)
                            input_mask = [1]*(len(part1+part2)+5+k)
                            n_pad = n_pad - 1
                            segment_ids.extend([0]*n_pad)
                            input_mask.extend([0]*n_pad)
                result_dict[str(i)] = (' '.join(output)).replace(' ##','').replace(' [SEP]','').replace(', ','')
            torch.save(model.state_dict(), os.path.join(save_dir, log_name+'_'+str(valid_loss)[:5]+'.pt'))
        else:
            stop += 1
            if stop > 10:
                break
    return result_dict

test_imag, test_ques, test_answ = extract_data(test_text_file,0,196)
mod_dict_op, mod_yn_dict_op = {value:key for key, value in mod_dict.items()}, {value:key for key, value in mod_yn_dict.items()}

mod_result, mod_loss = train_model(63,mod_model().to(device),mod_imag,mod_ques,mod_answ,mod_imag,mod_ques,mod_answ,'mod',6,0,72,False,mod_dict_op)
mod_yn_result, mod_yn_loss = train_model(40,mod_yn_model().to(device),mod_yn_imag,mod_yn_ques,mod_yn_answ,mod_yn_imag,mod_yn_ques,mod_yn_answ,'mod_yn',3,73,125,True,mod_yn_dict_op)
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
result_dict = dict(mod_result)
result_file = save_dir+'2019_result_modality.csv'

for i in tqdm(range(125)):
    fw.write(str(i+1))
    fw.write('	')
    fw.write(test_imag[i])
    fw.write('	')
    if i>=0 and i<72:
        fw.write([result_dict][mod_loss][str(i)])
    elif i>=73 and i<125:
        fw.write([result_yn_dict][mod_yn_loss][str(i)])
    else:
        fw.write([result_dict][0][str(i)])
    
        
    fw.write('\n')
fw.close()