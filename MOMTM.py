import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import braycurtis
from tqdm import tqdm  # 导入tqdm库
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy.spatial import distance
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr
import seaborn as sns
import ptitprince as pt
import random
from scipy.stats import gaussian_kde

pandas2ri.activate()
zCompositions = importr('zCompositions')
metadata_path = 'your_metadata.csv'
otu_data_path = 'your_data_file.csv'
metadata = pd.read_csv(metadata_path, encoding='GBK')
otu_data = pd.read_csv(otu_data_path, encoding='GBK')

# 重命名otu_data中的样本ID列，以便与metadata中的Sample_ID列匹配
otu_data.rename(columns={'Unnamed: 0': 'sample_ID'}, inplace=True)
#otu_data.iloc[:, 1:] = otu_data.iloc[:, 1:].div(otu_data.iloc[:, 1:].sum(axis=1), axis=0)
combined_data = pd.merge(metadata, otu_data, on='sample_ID', how='left')
combined_data = combined_data[combined_data['body_Site'] == 'Stool']
original_data = combined_data.copy()
#将样本信息转换为行名，利于后面的数据处理
def combine_sample_info(row):
    return '|'.join(map(str, row))

combined_data['sample_info'] = combined_data.iloc[:, :11].apply(combine_sample_info, axis=1)
# 将'sample_info'列设置为索引
combined_data.set_index('sample_info', inplace=True)

combined_data = combined_data[combined_data['body_Site'] == 'Stool']
# 删除前7列样本信息
combined_data = combined_data.iloc[:, 11:]


def filter_and_replace_zeros(df, threshold=0.5):
    # 计算每个特征中零值的比例
    zero_proportion = (df == 0).mean()
    # 保留零值比例小于阈值的特征
    filtered_df = df.loc[:, zero_proportion < threshold]
    # 确保数据类型为浮点数
    filtered_df = filtered_df.astype(float)
    # 将NA值替换为零
    filtered_df = filtered_df.fillna(0)
    # 将pandas数据帧转换为R数据帧
    r_df = pandas2ri.py2rpy(filtered_df)
    # 使用zCompositions包进行零替换
    r_result = zCompositions.cmultRepl(r_df, label=0, method="CZM")
    # 将结果转换回pandas数据帧
    replaced_df = pandas2ri.rpy2py(r_result)
    return replaced_df
combined_data_replaced = filter_and_replace_zeros(combined_data)
#这部分为了feature移除的时候，运行filter_and_replace_zeros后features数量一致
combined_data_replaced_features = combined_data_replaced.copy()
combined_data_replaced_features['sample_ID'] = combined_data_replaced_features.index.str.split('|').str[0]
combined_data_replaced_features['stage'] = combined_data_replaced_features.index.str.split('|').str[1]
combined_data_replaced_features['generations'] = combined_data_replaced_features.index.str.split('|').str[2]
combined_data_replaced_features['Breed'] = combined_data_replaced_features.index.str.split('|').str[3]
combined_data_replaced_features['pair_ID'] = combined_data_replaced_features.index.str.split('|').str[4]
combined_data_replaced_features['pig_ID'] = combined_data_replaced_features.index.str.split('|').str[5]
combined_data_replaced_features['area'] = combined_data_replaced_features.index.str.split('|').str[6]
combined_data_replaced_features['day'] = combined_data_replaced_features.index.str.split('|').str[7]
combined_data_replaced_features['group'] = combined_data_replaced_features.index.str.split('|').str[8]
combined_data_replaced_features['group_detail'] = combined_data_replaced_features.index.str.split('|').str[9]
combined_data_replaced_features['body_site'] = combined_data_replaced_features.index.str.split('|').str[10]
combined_data_replaced_features['day'] = pd.to_numeric(combined_data_replaced_features['day'], errors='coerce')
def clr_transformation(df):
    geometric_mean = df.apply(lambda x: np.exp(np.mean(np.log(x + 1e-10))), axis=1)
    clr_df = np.log(df.div(geometric_mean, axis=0) + 1e-10)
    return clr_df

combined_data_clr = clr_transformation(combined_data_replaced)
combined_data_clr['sample_ID'] = combined_data_clr.index.str.split('|').str[0]
combined_data_clr['stage'] = combined_data_clr.index.str.split('|').str[1]
combined_data_clr['generations'] = combined_data_clr.index.str.split('|').str[2]
combined_data_clr['Breed'] = combined_data_clr.index.str.split('|').str[3]
combined_data_clr['pair_ID'] = combined_data_clr.index.str.split('|').str[4]
combined_data_clr['pig_ID'] = combined_data_clr.index.str.split('|').str[5]
combined_data_clr['area'] = combined_data_clr.index.str.split('|').str[6]
combined_data_clr['day'] = combined_data_clr.index.str.split('|').str[7]
combined_data_clr['group'] = combined_data_clr.index.str.split('|').str[8]
combined_data_clr['group_detail'] = combined_data_clr.index.str.split('|').str[9]
combined_data_clr['body_site'] = combined_data_clr.index.str.split('|').str[10]
combined_data_clr['day'] = pd.to_numeric(combined_data_clr['day'], errors='coerce')
sow_data = combined_data_clr[combined_data_clr['generations'] == 'sow']
sow_data = sow_data.dropna(subset=sow_data.columns[10:])
sow_data_features = combined_data_replaced_features[combined_data_replaced_features["generations"] == 'sow']
sow_data_features = sow_data_features.dropna(subset=sow_data_features.columns[11:])
#print( np.unique(sow_data["stage"])) ['G85''G100''G112''S_D1''S_D3''S_D7''S_D14''S_D21'
 #  ]
piglet_data = combined_data_clr[combined_data_clr['generations'] == 'piglet']
piglet_data_features = combined_data_replaced_features[combined_data_replaced_features["generations"] == 'piglet']
piglet_data = piglet_data.dropna(subset=piglet_data.columns[10:])
piglet_data_features = piglet_data_features.dropna(subset=piglet_data_features.columns[10:])
#分割piglet_data为两个部分
# 从 'stage' 列提取天数，使用 raw string notation 避免 invalid escape sequence
#piglet_data['day'] = piglet_data['stage'].str.extract(r'D(\d+)').astype(int)
piglet_data['day'] = pd.to_numeric(piglet_data['day'], errors='coerce')
piglet_data_features['day'] = pd.to_numeric(piglet_data_features['day'], errors='coerce')
sow_data['day'] = pd.to_numeric(sow_data['day'], errors='coerce')
sow_data_features['day'] = pd.to_numeric(sow_data_features['day'], errors='coerce')
# 根据天数分割数据为 sucking 和 weaned
filter_train_data = piglet_data[piglet_data['group_detail'].isin(['Local', 'Commercial'])]
filter_train_data2  = piglet_data[~piglet_data['group_detail'].isin(['Local', 'Commercial'])]
unique_NX_pair_ids = filter_train_data2['pair_ID'].unique()
train_NX_pair_ids, test_NX_pair_ids = train_test_split(unique_NX_pair_ids, test_size=0.1, random_state=42)
train_NX_data = filter_train_data2[filter_train_data2['pair_ID'].isin(train_NX_pair_ids)]
filter_train_data = pd.concat([train_NX_data, filter_train_data], ignore_index=True)
#filter_train_data = piglet_data
filter_train_data_features = piglet_data_features[piglet_data_features['group_detail'].isin(['Local', 'Commercial'])]
filter_train_data_features2 = piglet_data_features[~piglet_data_features['group_detail'].isin(['Local', 'Commercial'])]
train_NX_data_feature = filter_train_data_features2[filter_train_data_features2['pair_ID'].isin(train_NX_pair_ids)]
filter_train_data_features = pd.concat([train_NX_data_feature, filter_train_data_features], ignore_index=True)
#filter_train_data_features = piglet_data_features
sucking_data = filter_train_data[filter_train_data['day'] <= 21]
sucking_data_features =filter_train_data_features[filter_train_data_features['day'] <= 21]


#print( np.unique(sucking_data["stage"]))
#['D1' 'D3' 'D7''D10' 'D14' 'D18' 'D21' 'D28' ]
weaned_data = piglet_data[piglet_data['day'] > 21]
weaned_data_features =piglet_data_features[piglet_data_features['day'] > 21]
#print( np.unique(weaned_data["stage"]))
#['D35''D49''D63''D70''D90''D120''D150']
#计算哺乳期：
unique_sucking_pair_ids = sucking_data_features['pair_ID'].unique()
train_pair_ids, test_pair_ids = train_test_split(unique_sucking_pair_ids, test_size=0.3, random_state=42)
# 根据 pair_IDs 分割 sow_data 和 piglet_data
train_sucking_data = sucking_data[sucking_data['pair_ID'].isin(train_pair_ids)]
test_sucking_data = sucking_data[sucking_data['pair_ID'].isin(test_pair_ids)]


def fill_and_mask(data, stages):
    mask_df = pd.DataFrame()
    filled_data = pd.DataFrame()
    for pig_id in data['pig_ID'].unique():
        pig_data = data[data['pig_ID'] == pig_id]

        mask = {stage: False for stage in stages}
        existing_stages = pig_data['stage'].unique()

        for stage in existing_stages:
            mask[stage] = True

        mask_record = {'pig_ID': pig_id if pd.notna(pig_id) else 'NaN'}
        mask_record.update(mask)
        mask_df = pd.concat([mask_df, pd.DataFrame([mask_record])], ignore_index=True)

        for stage in stages:
            if stage not in existing_stages:
                if not pig_data.empty:
                    sample_data = pig_data.iloc[0].copy()
                    sample_data['stage'] = stage
                    filled_data = pd.concat([filled_data, pd.DataFrame([sample_data])], ignore_index=True)
            else:
                stage_data = pig_data[pig_data['stage'] == stage]
                filled_data = pd.concat([filled_data, stage_data], ignore_index=True)

    return filled_data, mask_df

# Example usage with your data

sow_stages = ['G85', 'G100', 'G109', 'G112', 'S_D1', 'S_D3', 'S_D7', 'S_D14', 'S_D21']
#filled_sow_data = fill_missing_stages(sow_data, sow_stages)
sucking_stages = ['D3', 'D7', 'D14', 'D21']
weaned_stages = ['D35', 'D49', 'D63', 'D70', 'D90', 'D120', 'D150']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
filled_sucking_data, masks = fill_and_mask(train_sucking_data, sucking_stages)
filled_sucking_data_features, masks_features = fill_and_mask(sucking_data_features, sucking_stages)

filled_test_sucking_data, mask_df = fill_and_mask(test_sucking_data, sucking_stages)

def euclidean_loss(pred, target, masks):
    target_masked = target * masks.unsqueeze(-1)
    pred_masked = pred * masks.unsqueeze(-1)
    diff_squared = (pred_masked - target_masked) ** 2
    sum_diff_squared = torch.sum(diff_squared, dim=2)
    distance_sum = torch.sum(sum_diff_squared * masks, dim=1)
    valid_elements = torch.sum(masks, dim=1)
    loss_euclidean = distance_sum / valid_elements
    return torch.sqrt(loss_euclidean.mean())

def mse_loss(pred, target, masks):
    pred_masked = pred * masks.unsqueeze(-1)
    target_masked = target * masks.unsqueeze(-1)
    mse = F.mse_loss(pred_masked, target_masked, reduction='none')
    loss_mse = torch.sum(mse * masks.unsqueeze(-1), dim=[1, 2]) / torch.sum(masks, dim=1, keepdim=True)
    return loss_mse.mean()

class CompositeLossWithL1(nn.Module):
    def __init__(self, alpha=0, epsilon=1, l1_lambda=0):
        super(CompositeLossWithL1, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.l1_lambda = l1_lambda
        self.eps = 1e-10  # Adding epsilon to avoid division by zero

    def forward(self, pred, target, masks, model):
        target_masked = target * masks.unsqueeze(-1)
        pred_masked = pred * masks.unsqueeze(-1)
        mse_loss_value = F.mse_loss(pred_masked, target_masked, reduction='none')
        mse_loss_value = torch.sum(mse_loss_value * masks.unsqueeze(-1), dim=[1,2]) / (torch.sum(masks) + self.eps)
        mse_loss_value = mse_loss_value.mean()
        loss_euclidean = euclidean_loss(pred, target, masks)
        l1_loss = sum(p.abs().sum() for p in model.parameters())
        total_loss = (self.alpha * loss_euclidean + self.epsilon * mse_loss_value + self.l1_lambda * l1_loss)
        return total_loss

class MicrobiomeDataset(Dataset):
    def __init__(self, sow_data, piglet_data, mask_df, stages):
        self.sow_data = sow_data
        self.piglet_data = piglet_data
        self.mask_df = mask_df
        self.stages = stages
        self.pig_ids = piglet_data['pig_ID'].unique()
        self.sample_pairs = pd.DataFrame(columns=['sow_sample_ID', 'piglet_sample_ID'])

    def __len__(self):
        return len(self.pig_ids)

    def __getitem__(self, idx):
        pig_id = self.pig_ids[idx]
        piglet_tensors = []
        sow_tensors = []
        mask_row = self.mask_df[self.mask_df['pig_ID'] == pig_id]
        if mask_row.empty:
            mask_values = [False] * len(self.stages)
        else:
            mask_values = list(mask_row.iloc[0][1:])
        mask_tensor = torch.tensor(mask_values, dtype=torch.float32)
        for stage in self.stages:
            stage_piglet_data = self.piglet_data[(self.piglet_data['pig_ID'] == pig_id) & (self.piglet_data['stage'] == stage)]
            if stage_piglet_data.empty:
                piglet_tensor = torch.zeros((1, self.piglet_data.shape[1] - 11), dtype=torch.float32)
            else:
                piglet_tensor = torch.tensor(stage_piglet_data.iloc[:, :-11].values, dtype=torch.float32)
            piglet_tensors.append(piglet_tensor)
            sow_sample_ids = []
            for _, row in stage_piglet_data.iterrows():
                pair_id = row['pair_ID']
                piglet_day = row['day']
                sow_subset = self.sow_data[self.sow_data['pair_ID'].isin([pair_id])]
                sow_data = sow_subset.iloc[sow_subset['day'].sub(piglet_day).abs().argsort()[:1]]
                if not sow_data.empty:
                    sow_sample_ids.extend(sow_data['sample_ID'].values)
                    self.sample_pairs = self.sample_pairs._append({
                        'sow_sample_ID': sow_data['sample_ID'].iloc[0],
                        'piglet_sample_ID': row['sample_ID']
                    }, ignore_index=True)
            if sow_sample_ids:
                matched_sow_data = self.sow_data[self.sow_data['sample_ID'].isin(sow_sample_ids)]
                sow_tensor = torch.tensor(matched_sow_data.iloc[:, :-11].values, dtype=torch.float32)
            else:
                sow_tensor = torch.zeros_like(piglet_tensor)
            sow_tensors.append(sow_tensor)
        piglet_tensor_stack = torch.stack(piglet_tensors)
        sow_tensor_stack = torch.stack(sow_tensors)
        piglet_tensor_stack = torch.squeeze(piglet_tensor_stack, 1)
        sow_tensor_stack = torch.squeeze(sow_tensor_stack, 1)
        return piglet_tensor_stack, sow_tensor_stack, mask_tensor

def enforce_non_negative_weights(model):
    with torch.no_grad():
        for param in model.parameters():
            param.clamp_(min=0)

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, encoder_outputs, hidden):
        timestep = encoder_outputs.size(1)
        hidden = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        attn_energies = torch.tanh(self.attn(encoder_outputs))
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(2)
        attn_weights = torch.bmm(attn_energies, v).squeeze(2)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=1).unsqueeze(1)
        return attn_weights

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_steps):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_steps = num_steps
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, hidden, cell):
        decoder_input = torch.zeros((hidden.size(1), self.num_steps, self.hidden_size)).to(hidden.device)
        outputs, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))
        attn_weights = self.attention(encoder_outputs, hidden[-1])
        context = torch.bmm(attn_weights, encoder_outputs).squeeze(1)
        outputs = outputs + context.unsqueeze(1).repeat(1, self.num_steps, 1)
        predictions = self.fc(outputs)
        return predictions

class EncoderDecoderModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input):
        encoder_outputs, hidden, cell = self.encoder(encoder_input)
        decoder_output = self.decoder(encoder_outputs, hidden, cell)
        return decoder_output

# Model parameters
input_size = "your_input_size"
hidden_size = 50
output_size = 65
num_steps = "your_num_steps"

encoder = Encoder(input_size, hidden_size).to(device)
decoder = Decoder(hidden_size, output_size, num_steps).to(device)
model = EncoderDecoderModel(encoder, decoder).to(device)

# Dataset and DataLoader
dataset = MicrobiomeDataset(sow_data, filled_sucking_data, masks, sucking_stages)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
dataset_test = MicrobiomeDataset(sow_data, filled_test_sucking_data, mask_df, sucking_stages)
dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=True)

# Loss function and optimizer
loss_fn = CompositeLossWithL1(alpha=1, epsilon=1, l1_lambda=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from epoch {epoch} with loss {loss}")
    return model, optimizer, epoch, loss

train_losses = []
test_losses = []
train_mse = []
train_mae = []
train_r2 = []
num_epochs = 150

best_loss = float('inf')
best_model_path = 'your_output_path'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for epoch in range(num_epochs):
    train_loss = 0.0
    mse_scores = []
    mae_scores = []
    r2_scores = []

    model.train()

    for sow_data1, piglet_data, masks1 in dataloader:
        sow_data1 = sow_data1.to(device)
        piglet_data = piglet_data.to(device)
        masks1 = masks1.to(device)
        masks2 = masks1.unsqueeze(-1)

        piglet_data_masked = piglet_data * masks2

        optimizer.zero_grad()
        batch_size, time_steps, feature_size = sow_data1.shape

        encoder_input = sow_data1.view(batch_size, time_steps, feature_size)

        # 直接预测所有时间点的数据
        outputs = model(encoder_input)
        loss = loss_fn(outputs, piglet_data_masked, masks1, model)

        loss.backward()
        optimizer.step()

        # 应用非负权重约束
        enforce_non_negative_weights(model)

        train_loss += loss.item()

        true_np = piglet_data.view(-1, feature_size).detach().cpu().numpy()
        pred_np = outputs.view(-1, feature_size).detach().cpu().numpy()

        mse = mean_squared_error(true_np, pred_np)
        mae = mean_absolute_error(true_np, pred_np)
        r2 = r2_score(true_np, pred_np)

        mse_scores.append(mse)
        mae_scores.append(mae)
        r2_scores.append(r2)

    epoch_train_loss = train_loss / len(dataloader)
    epoch_mse = np.mean(mse_scores)
    epoch_mae = np.mean(mae_scores)
    epoch_r2 = np.mean(r2_scores)

    train_losses.append(epoch_train_loss)
    train_mse.append(epoch_mse)
    train_mae.append(epoch_mae)
    train_r2.append(epoch_r2)

    print(f'Epoch {epoch + 1}, Train Loss: {epoch_train_loss}, MSE: {epoch_mse}, MAE: {epoch_mae}, R²: {epoch_r2}')
    scheduler.step(epoch_train_loss)

    if epoch_train_loss < best_loss:
        best_loss = epoch_train_loss
        save_checkpoint(model, optimizer, epoch, best_loss, best_model_path)

plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(train_mse, label='Train MSE', color='r')
plt.title('Training MSE over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(train_mae, label='Train MAE', color='g')
plt.title('Training MAE over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(train_r2, label='Train R²', color='b')
plt.title('Training R² over Epochs')
plt.xlabel('Epoch')
plt.ylabel('R²')
plt.legend()

plt.tight_layout()
plt.show()

#model, optimizer, epoch, loss = load_checkpoint(model, optimizer, 'H:/新建文件夹/25.博士论文参考文献/博士毕业数据/4.15.2024数据处理/attention+weight_loss_postive_rate.pt')


def plot_with_fit(x, y, xlabel, ylabel, title, color, label):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c=color, label=label)

    # Linear fit
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs
    y_fit = np.polyval(coeffs, x)
    r2 = r2_score(y, y_fit)

    # Plot linear fit
    plt.plot(x, y_fit, color='black', linestyle='--', label=f'Fit: y={slope:.2f}x+{intercept:.2f}, R²={r2:.2f}')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def evaluate_model(dataloader, model, device):
    model.eval()
    mse_scores = []
    mae_scores = []
    r2_scores = []  # 初始化 R² 分数列表
    all_true_values = []
    all_predicted_values = []

    with torch.no_grad():
        for sow_data1, piglet_data, masks1 in dataloader:
            sow_data1 = sow_data1.to(device)
            piglet_data = piglet_data.to(device)
            masks2 = masks1.unsqueeze(-1).to(device)
            masks1 = masks1.unsqueeze(-1).to(device)
            piglet_data_masked = piglet_data * masks2
            batch_size, seq_length, feature_size = sow_data1.shape
            encoder_input = sow_data1.view(batch_size, seq_length, feature_size)
            # 直接预测所有时间点的数据
            outputs = model(encoder_input) * masks1
            true_values = piglet_data_masked.view(-1, feature_size).detach().cpu().numpy()
            predicted_values = outputs.view(-1, feature_size).detach().cpu().numpy()
            if true_values.size > 0 and predicted_values.size > 0:
                all_true_values.append(true_values)
                all_predicted_values.append(predicted_values)

            for feature_index in range(feature_size):
                true_feature_values = true_values[:, feature_index].flatten()
                predicted_feature_values = predicted_values[:, feature_index].flatten()

                mse = mean_squared_error(true_feature_values, predicted_feature_values)
                mae = mean_absolute_error(true_feature_values, predicted_feature_values)
                r2 = r2_score(true_feature_values, predicted_feature_values)  # 计算 R² 值

                mse_scores.append(mse)
                mae_scores.append(mae)
                r2_scores.append(r2)  # 添加 R² 分数到列表

    all_true_values = np.vstack(all_true_values)
    all_predicted_values = np.vstack(all_predicted_values)

    # 使用PCA进行降维
    pca = PCA(n_components=2)
    all_data = np.vstack([all_true_values, all_predicted_values])
    pca_transformed = pca.fit_transform(all_data)
    true_values_transformed = pca_transformed[:len(all_true_values)]
    predicted_values_transformed = pca_transformed[len(all_true_values):]

    plt.figure(figsize=(10, 6))
    plt.scatter(true_values_transformed[:, 0], true_values_transformed[:, 1], c='#7c9d97', label='True Values')
    plt.scatter(predicted_values_transformed[:, 0], predicted_values_transformed[:, 1], c='#e9b383',
                label='Predicted Values')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Comparison of True and Predicted Values on PC1 and PC2")
    plt.grid(True)
    plt.legend()
    plt.show()

    plot_with_fit(true_values_transformed[:, 0], predicted_values_transformed[:, 0],
                  "True Values PC1", "Predicted Values PC1", "True Values vs Predicted Values on PC1",
                  '#7c9d97', 'PC1')

    plot_with_fit(true_values_transformed[:, 1], predicted_values_transformed[:, 1],
                  "True Values PC2", "Predicted Values PC2", "True Values vs Predicted Values on PC2",
                  '#e9b383', 'PC2')

    avg_mse = np.mean(mse_scores)
    avg_mae = np.mean(mae_scores)
    avg_r2 = np.mean(r2_scores)

    return avg_mse, avg_mae, avg_r2


# 示例用法，确保 dataloader_test, model, device 已定义
mse, mae, r2 = evaluate_model(dataloader_test, model, device)
print(f"Test Average MSE: {mse}, Test Average MAE: {mae}, Test Average R²: {r2}")

feature_names = sucking_data_features.columns[:-11]
def zero_replacement(df):
    df = df.astype(float)
    r_df = pandas2ri.py2rpy(df)
    r_result = zCompositions.cmultRepl(r_df, label=0, method="CZM")
    replaced_df = pandas2ri.rpy2py(r_result)
    return replaced_df

def clr_transformation(df):
    geometric_mean = df.apply(lambda x: np.exp(np.mean(np.log(x + 1e-10))), axis=1)
    clr_df = np.log(df.div(geometric_mean, axis=0) + 1e-10)
    return clr_df

def preprocess_data(data, removed_feature):
    modified_data = data.copy()
    if removed_feature in modified_data.columns:
        modified_data[removed_feature] = 0
    # 归一化数据
    modified_data = modified_data.div(modified_data.sum(axis=1), axis=0)
    return modified_data


def get_tensors(sow_data, piglet_data, mask_df, stages, feature_names):
    pig_ids = piglet_data['pig_ID'].unique()
    piglet_tensors = {}
    sow_tensors_before = {}
    sow_tensors_after = {}
    relative_abundances = {}

    for pig_id in tqdm(pig_ids, desc="Processing piglets"):
        mask_row = mask_df[mask_df['pig_ID'] == pig_id]
        mask_values = [False] * len(stages) if mask_row.empty else [mask_row.iloc[0].get(stage, False) for stage in stages]
        mask_tensor = torch.tensor(mask_values, dtype=torch.float32)
        stage_piglet_data1 = pd.DataFrame()
        for stage in stages:
            stage_piglet_data = piglet_data[(piglet_data['pig_ID'] == pig_id) & (piglet_data['stage'] == stage)]
            if stage_piglet_data.empty:
                continue
            stage_piglet_data1 = pd.concat([stage_piglet_data1, stage_piglet_data])
            stage_piglet_data = stage_piglet_data.iloc[:, :-11]
            stage_piglet_data = clr_transformation(stage_piglet_data)
            piglet_tensor = torch.tensor(stage_piglet_data.values, dtype=torch.float32)
            piglet_tensors[(pig_id, stage)] = piglet_tensor

            for _, row in stage_piglet_data1.iterrows():
                pair_id = row['pair_ID']
                sow_subset = sow_data[sow_data['pair_ID'] == pair_id]
                closest_sow_data = sow_subset.iloc[(sow_subset['day'] - row['day']).abs().argsort()[:1]]

                if not closest_sow_data.empty:
                    closest_sow_data = closest_sow_data.iloc[:, :-11]
                    closest_sow_data_clr = clr_transformation(closest_sow_data)
                    sow_tensor_before = torch.tensor(closest_sow_data_clr.values, dtype=torch.float32)
                    sow_tensors_before[(pig_id, stage)] = sow_tensor_before

                    for removed_feature in feature_names:
                        relative_abundance = closest_sow_data[removed_feature].values[0]
                        modified_sow_data = preprocess_data(closest_sow_data.copy(), removed_feature)
                        modified_sow_data_clr = clr_transformation(modified_sow_data)
                        sow_tensor_after = torch.tensor(modified_sow_data_clr.values, dtype=torch.float32)
                        sow_tensors_after[(pig_id, stage, removed_feature)] = sow_tensor_after

                        # 存储相对丰度
                        relative_abundances[(pig_id, stage, removed_feature)] = relative_abundance

    return piglet_tensors, sow_tensors_before, sow_tensors_after, mask_tensor, relative_abundances

piglet_tensors, sow_tensors_before, sow_tensors_after, mask_tensor,relative_abundances = get_tensors(sow_data_features, filled_sucking_data_features,
                                                                                                     masks_features, sucking_stages, feature_names)

def euclidean_distance(piglet_tensor_batch, pred_batch):
    # 计算欧式距离
    diff = piglet_tensor_batch - pred_batch
    dist_squared = torch.sum(diff ** 2, dim=1)  # 沿最后一个维度求和
    euclidean_dist = torch.sqrt(dist_squared)

    # 返回批次的平均欧式距离
    return euclidean_dist.mean().item()


def calculate_transmission_coreness(model, piglet_tensors, sow_tensors_before, sow_tensors_after,
                                    relative_abundances, feature_names, stages):
    model.eval()
    bc_distances = []
    device = next(model.parameters()).device  # 获取模型的设备
    with torch.no_grad():
        # 按照 pig_id 进行分组
        pig_ids = set(pig_id for pig_id, _ in piglet_tensors.keys())
        for pig_id in tqdm(pig_ids, desc="Calculating coreness"):
            # 获取该 pig_id 对应的所有阶段的 piglet_tensor 和 sow_tensors_before
            try:
                piglet_tensors_for_pig = [piglet_tensors[(pig_id, stage)].to(device) for stage in stages]
                sow_tensors_before_for_pig = [sow_tensors_before[(pig_id, stage)].to(device) for stage in stages]
            except KeyError as e:
                print(f"KeyError: {e} - missing data for pig_id: {pig_id} in stages")
                continue
            for removed_feature in feature_names:
                # 获取该 pig_id 对应的所有阶段的 sow_tensors_after
                sow_tensors_after_for_pig = []
                valid = True
                for stage in stages:
                    if (pig_id, stage, removed_feature) in sow_tensors_after:
                        sow_tensors_after_for_pig.append(sow_tensors_after[(pig_id, stage, removed_feature)].to(device))
                    else:
                        valid = False
                        break
                if not valid:
                    continue
                # 转换为张量
                sow_tensor_before_batch = torch.stack(sow_tensors_before_for_pig)
                sow_tensor_after_batch = torch.stack(sow_tensors_after_for_pig)
                piglet_tensor_batch = torch.stack(piglet_tensors_for_pig)
                sow_tensor_before_batch = sow_tensor_before_batch.squeeze(1)
                sow_tensor_after_batch = sow_tensor_after_batch.squeeze(1)
                piglet_tensor_batch = piglet_tensor_batch.squeeze(1)
                # 模型预测
                combined_batch = torch.cat((sow_tensor_before_batch.unsqueeze(0), sow_tensor_after_batch.unsqueeze(0)),
                                           dim=0)
                # 打印检查
                # 计算欧式距离
                preds = model(combined_batch)
                pred_before, pred_after = preds[0], preds[1]
                dist_before = euclidean_distance(piglet_tensor_batch, pred_before)
                dist_after = euclidean_distance(piglet_tensor_batch, pred_after)
                # 获取相对丰度并计算差异
                for stage in stages:
                    if (pig_id, stage, removed_feature) in relative_abundances:
                        relative_abundance = relative_abundances[(pig_id, stage, removed_feature)]
                        dist_diff = abs(dist_after - dist_before)
                        # 保存结果
                        bc_distances.append({
                            'pig_id': pig_id,
                            'stage': stage,
                            'removed_feature': removed_feature,
                            'distance_before': dist_before,
                            'distance_after': dist_after,
                            'distance_diff': dist_diff
                        })
                    else:
                        print(f"Key {(pig_id, stage, removed_feature)} not found in relative_abundances")
    return bc_distances




# 确保传递的所有参数
bc_distances = calculate_transmission_coreness(
    model,
    piglet_tensors,
    sow_tensors_before,
    sow_tensors_after,
    relative_abundances,
    feature_names,
    sucking_stages
)
df = pd.DataFrame(bc_distances)
#用掩码去掉多余的数据
masks_features.set_index('pig_ID', inplace=True)
#filled_sucking_data_features, masks_features = fill_and_mask(piglet_data_features, sucking_stages)

# 定义一个函数来检查是否应该保留一行
def should_keep_row(row):
    pig_id = row['pig_id']
    stage = row['stage']
    if pig_id in masks_features.index and stage in masks_features.columns:
        return masks_features.at[pig_id, stage]
    return True  # 如果没有匹配的掩码信息，则保留该行

# 过滤 df 数据集
filtered_df = df[df.apply(should_keep_row, axis=1)]

metadata1 =metadata.copy()
metadata1.rename(columns={'pig_ID': 'pig_id'}, inplace=True)
merged_df = pd.merge(filtered_df, metadata1[['pig_id', 'Breed']], on='pig_id', how='left')
# 删除 pig_id、stage 和 removed_feature 都相同的重复行，只保留第一次出现的行
merged_df = merged_df.drop_duplicates(subset=['pig_id', 'stage', 'removed_feature'])
