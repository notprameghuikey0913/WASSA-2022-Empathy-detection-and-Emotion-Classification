import torch
import torch.nn as nn
from torch import cuda
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
scaler = MinMaxScaler()
PATH = 'wassa_model.pth'

device = 'cuda' if cuda.is_available() else 'cpu'

training_data = pd.read_csv("messages_train_ready_for_WS.tsv", sep='\t')
data1 = training_data[['essay', 'empathy', 'distress']]
data2 = training_data[['gender', 'education', 'race', 'age', 'income']]

data2 = pd.DataFrame(scaler.fit_transform(data2), columns=['gender', 'education', 'race', 'age', 'income'])
data1 = pd.concat([data1, data2], axis=1)

MAX_LEN = 256
TRAIN_BATCH_SIZE = 8

# EPOCHS = 1

LEARNING_RATE = 1e-05
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation = True, do_lower_case = True)

class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.essay
        self.targets = self.data.empathy
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
            'targets': torch.tensor(self.targets[index], dtype=torch.float),
            'gender': torch.tensor(self.gender[index], dtype=torch.float),
            'education': torch.tensor(self.education[index], dtype=torch.float),
            'race': torch.tensor(self.race[index], dtype=torch.float),
            'age': torch.tensor(self.age[index], dtype=torch.float),
            'income': torch.tensor(self.income[index], dtype=torch.float),
        }


train_size = 1.0
train_data = data1.sample(frac=train_size,random_state=200)
train_data = train_data.reset_index(drop=True)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }


training_set = SentimentData(train_data, tokenizer, MAX_LEN)
training_loader = DataLoader(training_set, **train_params)
training_loader2 = DataLoader(train_data, **train_params)

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

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = LEARNING_RATE)

def calculate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()

    for _, data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)
        targets = targets.view(-1, 1)
        gender = data['gender'].to(device, dtype = torch.float).view(-1, 1)
        education = data['education'].to(device, dtype = torch.float).view(-1, 1)
        race = data['race'].to(device, dtype = torch.float).view(-1, 1)
        age = data['age'].to(device, dtype = torch.float).view(-1, 1)
        income = data['income'].to(device, dtype = torch.float).view(-1, 1)

        outputs = model(ids, mask, token_type_ids, gender, education, race, age, income)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        n_correct += calculate_accuracy(outputs, targets)

        nb_tr_steps +=1
        nb_tr_examples+=targets.size(0)

        if _%50==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples
            writer.add_scalar('training loss', loss_step, nb_tr_examples)
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return

EPOCHS = 1

for epoch in range(EPOCHS):
    train(epoch)

torch.save(model.state_dict(), PATH)

print('All files saved')

