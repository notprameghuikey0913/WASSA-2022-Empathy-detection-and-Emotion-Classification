import torch
import torch.nn as nn
from torch import cuda
import math
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import transformers
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
writer = SummaryWriter("runs/wassa")
scaler = StandardScaler()
scaler2 = MinMaxScaler()

device = 'cuda' if cuda.is_available() else 'cpu'
PATH = 'wassa_model.pth'

training_data = pd.read_csv("messages_dev_features_ready_for_WS_2022.tsv", sep='\t')
target_data = pd.read_csv("goldstandard_dev_2022.tsv", sep='\t', header=None)
data1 = training_data[['essay']]
data2 = training_data[['gender', 'education', 'race', 'age', 'income']]
data3 = target_data[[0]]
empathy = data3.reset_index(drop = True)
targets = torch.tensor(empathy.to_numpy(), dtype=torch.float32)

data2 = pd.DataFrame(scaler.fit_transform(data2), columns=['gender', 'education', 'race', 'age', 'income'])
data1 = pd.concat([data1, data2], axis=1)
to_round = 4

MAX_LEN = 256
TRAIN_BATCH_SIZE = 8

# EPOCHS = 1

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation = True, do_lower_case = True)

class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.essay
        self.gender = dataframe.gender
        self.education = dataframe.education
        self.age = dataframe.age
        self.race = dataframe.race
        self.income = dataframe.income
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'gender': torch.tensor(self.gender[index], dtype=torch.float),
            'education': torch.tensor(self.education[index], dtype=torch.float),
            'race': torch.tensor(self.race[index], dtype=torch.float),
            'age': torch.tensor(self.age[index], dtype=torch.float),
            'income': torch.tensor(self.income[index], dtype=torch.float),
        }


train_data = data1.reset_index(drop=True)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }


training_set = SentimentData(train_data, tokenizer, MAX_LEN)
training_loader = DataLoader(training_set, **train_params)

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(768, 512)
        self.pre_final = torch.nn.Linear(517, 256)
        self.final = torch.nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask, token_type_ids, gender, education, race, age, income):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = nn.ReLU()(output)
        extra_inputs = torch.cat([output, gender, education, race, age, income], 1)
        output = self.pre_final(extra_inputs)
        output = nn.ReLU()(output)
        output = self.final(output)
        return output

model = RobertaClass()
model.to(device)
model.load_state_dict(torch.load(PATH))
final = torch.empty((270, 1))

def pearson_accuracy(preds, targets):
    print(preds.shape)
    x = [float(k) for k in preds]
    y = [float(k) for k in targets]

    xm = sum(x) / len(x)
    ym = sum(y) / len(y)

    xn = [k - xm for k in x]
    yn = [k - ym for k in y]

    r = 0
    r_den_x = 0
    r_den_y = 0
    for xn_val, yn_val in zip(xn, yn):
        r += xn_val * yn_val
        r_den_x += xn_val * xn_val
        r_den_y += yn_val * yn_val

    r_den = math.sqrt(r_den_x * r_den_y)

    if r_den:
        r = r / r_den
    else:
        r = 0

    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    r = max(min(r, 1.0), -1.0)

    return round(r, to_round)


def test(epoch):
    model.eval()
    tr_steps = 0

    with torch.no_grad():
        for _, data in tqdm(enumerate(training_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            gender = data['gender'].to(device, dtype=torch.float).view(-1, 1)
            education = data['education'].to(device, dtype=torch.float).view(-1, 1)
            race = data['race'].to(device, dtype=torch.float).view(-1, 1)
            age = data['age'].to(device, dtype=torch.float).view(-1, 1)
            income = data['income'].to(device, dtype=torch.float).view(-1, 1)

            outputs = model(ids, mask, token_type_ids, gender, education, race, age, income)
            print(tr_steps*TRAIN_BATCH_SIZE, tr_steps*TRAIN_BATCH_SIZE + TRAIN_BATCH_SIZE)
            if(tr_steps*TRAIN_BATCH_SIZE + TRAIN_BATCH_SIZE <= 270):
                final[tr_steps*TRAIN_BATCH_SIZE : tr_steps*TRAIN_BATCH_SIZE + TRAIN_BATCH_SIZE, :] = outputs
            else:
                final[tr_steps*TRAIN_BATCH_SIZE:, :] = outputs
            tr_steps += 1

        print(pearson_accuracy(final, targets))
    return

EPOCHS = 1

for epoch in range(EPOCHS):
    test(epoch)

print('All files tested')

final_np = final.numpy()
final_df = pd.DataFrame(final_np)
final_df.to_csv("predictions_EMP.tsv", sep='\t')