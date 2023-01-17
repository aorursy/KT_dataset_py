import torch



device = torch.device("cuda" if torch.cuda.is_available else "cpu")



print(device)
## 1. Reading data into a pandas DataFrame, and inspecting the columns a bit



import pandas as pd



df = pd.read_csv("../input/train_electricity.csv")  # <-- only this is important



print("Dataset has", len(df), "entries.")



print(f"\n\t{'Column':20s} | {'Type':8s} | {'Min':12s} | {'Max':12s}\n")

for col_name in df.columns:

    col = df[col_name]

    print(f"\t{col_name:20s} | {str(col.dtype):8s} | {col.min():12.1f} | {col.max():12.1f}")
import numpy as np



print("You shold score better than:")

print("RMS distance between Production_MW and Consumption_MW")

np.sqrt(np.mean((df["Production_MW"] - df["Consumption_MW"]) ** 2))

## 2. Adding some datetime related features (DO NOT RUN THIS CELL MORE THAN ONCE!)



def add_datetime_features(df):

    features = ["Year", "Week", "Day", "Dayofyear", "Month", "Dayofweek",

                "Is_year_end", "Is_year_start", "Is_month_end", "Is_month_start",

                "Hour", "Minute", "Quarter"]

    one_hot_features = ["Month", "Dayofweek", "Quarter"]

    datetime = pd.to_datetime(df.Date * (10 ** 9))



    df['Datetime'] = datetime  # <-- We won't use this for training, but we'll remove it later



    for feature in features:

        new_column = getattr(datetime.dt, feature.lower())

        if feature in one_hot_features:

            df = pd.concat([df, pd.get_dummies(new_column, prefix=feature)], axis=1)

        else:

            df[feature] = new_column

            

    return df



df = add_datetime_features(df)



print(f"{len(df.columns):d} columns:", ",".join([str(c) for c in df.columns]))
# Drop some columns



to_drop = ['Coal_MW', 'Gas_MW', 'Hidroelectric_MW', 'Nuclear_MW', 'Wind_MW', 'Solar_MW', 'Biomass_MW']

df.drop(columns=to_drop, inplace=True)
## 3. Split data into train / validation (leaving the last six months for validation)



from dateutil.relativedelta import relativedelta

import numpy as np



eval_from = df['Datetime'].max() + relativedelta(months=-6)  # Here we set the 6 months threshold

train_df = df[df['Datetime'] < eval_from].copy()

valid_df = df[df['Datetime'] >= eval_from].copy()



print(f"Train data: {train_df['Datetime'].min()} -> {train_df['Datetime'].max()} | {len(train_df)} samples.")

print(f"Valid data: {valid_df['Datetime'].min()} -> {valid_df['Datetime'].max()} | {len(valid_df)} samples.")

      

target_col = "Consumption_MW"

to_drop = ["Date", "Datetime", target_col]



# Create torch tensors with inputs / labels for both train / validation 

      

train_data = torch.Tensor(train_df.drop(columns=to_drop).values.astype(np.float)).to(device)

valid_data = torch.Tensor(valid_df.drop(columns=to_drop).values.astype(np.float)).to(device)

train_labels = torch.Tensor(train_df[target_col].values[:, None].astype(np.float)).to(device)

valid_labels = torch.Tensor(valid_df[target_col].values[:, None].astype(np.float)).to(device)
## 4. Normalize features (except one-hot ones)  - uncomment if you want to normalize this



"""

for idx in range(train_data.size(1)):

    if (train_data[:, idx].min() < 0 or train_data[:, idx].max() > 1):

        mean, std = train_data[:, idx].mean(), train_data[:, idx].std()

        train_data[:, idx].add_(-mean).div_(std)

        valid_data[:, idx].add_(-mean).div_(std)

""";
## 5. Prepare a simple model



from torch import nn, optim

from torch.nn import functional as F



HISTORY = 10  # How many time steps to look back into the recent history y_t = f(x_t-H+1, x_t-H+2, ..., x_t)



nfeatures = train_data.size(1)





class FeedForward(nn.Module):

    

    def __init__(self):

        super(FeedForward, self).__init__()

        self.model = nn.Sequential(

            nn.Linear(nfeatures * HISTORY, 500),

            # nn.BatchNorm1d(800),

            nn.ReLU(),

            nn.Dropout(p=0.2),

            nn.Linear(500, 100),

            # nn.BatchNorm1d(200),

            nn.ReLU(),

            nn.Dropout(p=0.2),

            nn.Linear(100, 1),

        )

    def forward(self, x):

        x = x.view(x.size(0), nfeatures * HISTORY)

        y = self.model(x)

        return y + x[:, nfeatures * (HISTORY - 1):nfeatures * (HISTORY - 1) + 1]





class RNN(nn.Module):

    

    def __init__(self):

        super(RNN, self).__init__()



        self.hsize = hsz = 50

        self.lstm = nn.LSTMCell(nfeatures, hsz)

        self.head = nn.Sequential(nn.Linear(hsz, 10), nn.ReLU(), nn.Linear(10, 1))

        

    def forward(self, x):

        x = x.permute(1, 0, 2)  # we permute it here so we won't change much in the train loop

        hsz = self.hsize

        tsz, bsz, _ = x.shape

        hx = torch.zeros(bsz, hsz, device=x.device)

        cx = torch.zeros(bsz, hsz, device=x.device)

        cx[:, 0] = 1.0

        

        for i in range(tsz):

            hx, cx = self.lstm(x[i], (hx, cx))

        return self.head(hx) + x[-1, :, 0:1]





model = FeedForward().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
# Here we write a validation routine. Observation: `model`, `valid_data`, `valid_labels` are global variables.

# TODO: cache the concatenated inputs



VALID_BATCH_SIZE = 5000



def validate():

    model.eval()

    

    nvalid = len(valid_data) - HISTORY + 1  # This is the number of validation examples

    losses = []



    with torch.no_grad():

        for start_idx in range(0, nvalid, VALID_BATCH_SIZE):

            end_idx = min(nvalid, start_idx + VALID_BATCH_SIZE)

            idxs = torch.arange(start_idx, end_idx)

            all_idxs = (idxs.unsqueeze(1) + torch.arange(HISTORY).unsqueeze(0)).view(-1)

            data = valid_data.index_select(0, all_idxs.to(device)).view(-1, HISTORY, nfeatures)

            label = valid_labels[idxs + HISTORY - 1]

            losses.append(F.mse_loss(model(data), label, reduction="none"))

        return np.sqrt(torch.cat(tuple(losses), dim=0).mean().item())
## 5. Train the model. You should do better than this. :)



STEPS_NO = 10000

REPORT_FREQ = 250

BATCH_SIZE = 64



nexamples = len(train_data) - HISTORY + 1



train_losses = []

model.train()



for step in range(1, STEPS_NO + 1):

    # prepare batch: sample i=t-H+1 to concat x_{t-H+1} .. x_t for input, and y_t for label

    

    idxs = torch.randint(nexamples, (BATCH_SIZE, 1), device=device)

    all_idxs = (idxs + torch.arange(HISTORY, device=device).unsqueeze(0)).view(-1)

    

    data = train_data.index_select(0, all_idxs).view(BATCH_SIZE, HISTORY, nfeatures)

    label = train_labels.index_select(0, idxs.view(-1) + (HISTORY - 1))

    

    # optimize using current batch

    optimizer.zero_grad()

    loss = F.mse_loss(model(data), label)

    loss.backward()

    optimizer.step()

    train_losses.append(loss.item())

    

    # report and monitor training

    if step % REPORT_FREQ == 0:

        valid_loss = validate()

        print(f"Step {step:4d}: Train RMSE={np.sqrt(np.mean(train_losses)):7.2f} | Valid RMSE={valid_loss:7.2f}")

        train_losses.clear()

        model.train()

    