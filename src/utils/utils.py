import os

import glob
from glob import glob
import h5py
import natsort
import numpy as np
import pydicom
import random
import cv2
from typing import Generator, Tuple
from scipy import ndimage

from IPython.display import clear_output


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, random_split, Subset, ConcatDataset
from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit
import matplotlib.pyplot as plt

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_dir(path, exist_ok=True):
    if not os.path.exists(path): 
        os.makedirs(path, exist_ok=exist_ok)
        os.chmod(path, 0o777)

def globsort(path):
    return natsort.natsorted(glob.glob(path))

def listdir(path):
    return natsort.natsorted(os.listdir(path))

def minMax_scaling(x, lb=0, m=None, M=None):
    if m==None: m = x.min()
    if M==None: M = x.max()
    if lb==0:
        return (x-m)/(M-m)
    if lb==-1:
        return (2*x-(M+m))/(M-m)

def load_model(hp, epoch, fold=None):
    if epoch == 'best':
        model_path = f'{hp.path_model}/fold_{fold}/best.pt'
    else:
        if fold is not None:
            model_path = f'{hp.path_model}/fold_{fold}/model_{epoch}.pt'
        else:
            model_path = f'{hp.path_model}/model_{epoch}.pt'

    obj = torch.load(model_path)
    if isinstance(obj, dict):
        model = hp.model
        model.load_state_dict(obj)
    else:
        model = obj
    return model


# def get_split_idx(ratio=[0.6, 0.2, 0.2], shuffle=True, generate=False):
#     if generate: 
#         list_patient = listdir('../data')
#         print(len(list_patient))
#         train, valid = train_test_split(list_patient, train_size=ratio[0], shuffle=shuffle)
#         valid, test  = train_test_split(valid       , train_size=ratio[1]/(ratio[1]+ratio[2]), shuffle=shuffle)
#         train = natsort.natsorted(train)
#         valid = natsort.natsorted(valid)
#         test = natsort.natsorted(test)
        
#         split_idx = {'train':train, 
#                      'valid':valid, 
#                      'test':test}
        
#         torch.save(split_idx, './utils/split_idx.pt')
    
#     else:
#         split_idx = torch.load('./utils/split_idx.pt')
        
#     return split_idx

def data_split(dataset: Dataset, seed = 42):

    total_size = len(dataset)

    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size  

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)  
    )

    return train_dataset, val_dataset, test_dataset

def k_fold_split(
    dataset: Dataset,
    k: int = 5,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Generator[Tuple[int, Subset, Subset, Subset], None, None]:

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    indices = list(range(len(dataset)))

    for fold, (trainval_idx, test_idx) in enumerate(kf.split(indices)):
        test_dataset = Subset(dataset, test_idx)

        trainval_subset = Subset(dataset, trainval_idx)

        trainval_size = len(trainval_subset)
        val_size = int(val_ratio * trainval_size)
        train_size = trainval_size - val_size

        train_subset, val_subset = random_split(
            trainval_subset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )

        yield fold, train_subset, val_subset, test_dataset


def get_sample_patient_groups(dataset, context: int = 0):
    """dataset의 각 샘플이 어느 환자(그룹) 소속인지 문자열 배열로 반환.

    meta 경로가 '.../<병변그룹>/<환자폴더>/<phase>/xxxxx.dcm' 형태라는 걸 이용해서
    '<병변그룹>/<환자폴더>'를 group key로 쓴다. GroupKFold/GroupShuffleSplit로
    환자 단위 분할을 할 때 씀 (같은 환자의 슬라이스가 train/val/test에 동시에
    들어가는 걸 원천적으로 막기 위함).
    """
    groups = []
    for i in range(len(dataset)):
        meta_group = dataset[i][2]
        center_meta = meta_group[context]
        parts = center_meta.replace('\\', '/').split('/')
        patient_folder = parts[-3]
        lesion_group = parts[-4]
        groups.append(f'{lesion_group}/{patient_folder}')
    return np.array(groups)


def k_fold_split_grouped(
    dataset: Dataset,
    groups: np.ndarray,
    k: int = 5,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Generator[Tuple[int, Subset, Subset, Subset], None, None]:
    """환자 단위(GroupKFold)로 완전히 분리된 train/val/test를 만든다.

    동일 환자의 슬라이스(같은 환자의 다른 phase, 인접 슬라이스 포함)가
    train/val/test 중 두 곳 이상에 걸치는 일이 없도록 보장한다.
    """
    indices = np.arange(len(dataset))

    gkf = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for fold, (trainval_idx, test_idx) in enumerate(gkf.split(indices, groups=groups)):
        test_dataset = Subset(dataset, test_idx)

        trainval_groups = groups[trainval_idx]
        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_rel_idx, val_rel_idx = next(gss.split(trainval_idx, groups=trainval_groups))
        train_idx = trainval_idx[train_rel_idx]
        val_idx = trainval_idx[val_rel_idx]

        # 환자 단위 분리가 실제로 지켜졌는지 방어적으로 검증
        g_train, g_val, g_test = set(groups[train_idx]), set(groups[val_idx]), set(groups[test_idx])
        assert not (g_train & g_val), f'train/val 환자 누수: {g_train & g_val}'
        assert not (g_train & g_test), f'train/test 환자 누수: {g_train & g_test}'
        assert not (g_val & g_test), f'val/test 환자 누수: {g_val & g_test}'

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        yield fold, train_subset, val_subset, test_dataset


def get_loaders(train_subset, val_subset, test_subset, worker_init_fn=None, batch_size=32, num_workers=0, pin_memory=True, collate_fn=None, g = None):
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn, worker_init_fn=worker_init_fn, generator = g)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn, worker_init_fn=worker_init_fn, generator = g)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn, worker_init_fn=worker_init_fn, generator = g)
    return train_loader, val_loader, test_loader

def load_single_dicom_array(path):
    ds = pydicom.dcmread(path)
    pixel_array = ds.pixel_array.astype(np.float32)
    pixel_array = minMax_scaling(pixel_array)
    return pixel_array * 255

def expand_indices_grouped(valid_indices, context, max_index):
    grouped = []
    for idx in valid_indices:
        group = []
        for offset in range(-context, context + 1):
            target_idx = idx + offset
            if 0 <= target_idx < max_index:
                group.append(target_idx)
            else:
                group.append(None)
        grouped.append(group)
    return grouped


# class Dataset_Temp(Dataset):
#     def __init__(self, hp, phase='train'):
#         self.hp = hp
#         self.phase = phase
#         split_idx = get_split_idx()[phase]
        
#         self.list_data = [h5py.File(f'{hp.path_data}/{patient}/data.h5', 'r') for patient in split_idx]
#         self.list_n = [len(data['X']) for data in self.list_data]
#         self.N = len(self.list_data)
        
#     def __len__(self):
#         return self.N
    
#     def __getitem__(self, idx):
#         k = np.random.randint(self.list_n[idx])
#         X_data = torch.from_numpy(self.list_data[idx]['X'][k]).to(torch.float32).unsqueeze(0)
#         y_data = torch.from_numpy(self.list_data[idx]['y'][k]).to(torch.float32)
        
#         return X_data, y_data
    
# class Dataset_Temp_inference(Dataset):
#     def __init__(self, hp, patient):
#         self.hp = hp
        
#         self.data = h5py.File(f'{hp.path_data}/{patient}/data.h5', 'r')
#         self.X_data = self.data['X'][:]
#         self.y_data = self.data['y'][:]
#         self.N = len(self.X_data)
        
#     def __len__(self):
#         return self.N

#     def __getitem__(self, idx):
#         X_data = torch.from_numpy(self.X_data[idx]).to(torch.float32).unsqueeze(0)
#         y_data = torch.from_numpy(self.y_data[idx]).to(torch.float32)
        
#         return X_data, y_data
    
class MAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p, y, w=None):
        res = torch.abs(p-y).mean()
        
        return res

class ACC(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, p, y):
        p = (p>self.alpha).float()
        res = (p==y).float()
        res = res.mean()
        
        return res


class Precision(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, p, y):
        p = torch.sigmoid(p).contiguous()
        p_bin = (p > self.alpha).float()
        y_bin = (y > 0.5).float()

        tp = (p_bin * y_bin).sum()
        pp = p_bin.sum()

        if pp == 0:
            return torch.tensor(0.0, device=p.device)  

        return tp / (pp + 1e-8)

class Specificity(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, p, y):
        p = torch.sigmoid(p).contiguous()
        p_bin = (p > self.alpha).float()
        y_bin = (y > 0.5).float()

        neg = 1.0 - y_bin
        tn = ((1.0 - p_bin) * neg).sum()
        fp = (p_bin * neg).sum()
        denom = tn + fp

        if denom == 0:
            return torch.tensor(0.0, device=p.device)

        return tn / (denom + 1e-8)

class Recall(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, p, y):
        p = torch.sigmoid(p).contiguous()
        p_bin = (p > self.alpha).float()
        y_bin = (y > 0.5).float()

        tp = (p_bin * y_bin).sum()
        actual_positive = y_bin.sum()

        if actual_positive == 0:
            return torch.tensor(0.0, device=p.device) 

        return tp / (actual_positive + 1e-8)

class DiceScore(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p, y, eps=1e-8):
        p_bin = (torch.sigmoid(p) > 0.5).float()
        y_bin = (y > 0.5).float()
        intersection = (p_bin * y_bin).sum()
        union = p_bin.sum() + y_bin.sum()
        return (2. * intersection + eps) / (union + eps)

def es(loss, epoch, inverse=False):
    escounter = ['E', '|']
    loss = loss[1:epoch+1, 1]
    loss_best = torch.min(loss, axis=0)[0]
    loss_curr = loss[epoch-1]
    if inverse==False:
        T = loss_best>=loss_curr
    else:
        T = loss_best<=loss_curr
    T = [escounter[i] for i in T]
    
    return T

def get_ylim(loss, title, alpha=2, margin=0.1):
    m = loss.min()
    M = loss.max()
    a = loss.mean()
    
    if M-a>(a*alpha):
        M = a*alpha
        
    m = m*(1-margin)
    M = M*(1+margin)
    
    l = len(loss)//2
    
    if torch.sum(loss[0:10][0])>torch.sum(loss[-10:][0]): 
        plt.axhline(loss[:, 1].min(), color='black', linewidth=0.7, linestyle='--', alpha=0.7)
        epoch = torch.argmin(loss[:, 1])
        plt.axvline(epoch, color='r', linewidth=0.7, linestyle='--', alpha=0.7)
        plt.title(f'{title}: {loss[:, 1].min():0.4f} ({loss[l:, 1].mean():0.4f})')
    else:
        plt.axhline(loss[:, 1].max(), color='black', linewidth=0.7, linestyle='--', alpha=0.7)
        epoch = torch.argmax(loss[:, 1])
        plt.axvline(epoch, color='r', linewidth=0.7, linestyle='--', alpha=0.7)
        plt.title(f'{title}: {loss[:, 1].max():0.4f} ({loss[l:, 1].mean():0.4f})')

    return m, M

def plot_history(hp, epoch=0, num_fold=None, loss_titles=None, rate_titles=None):
    if num_fold is not None:
        path = f'{hp.path_model}/fold_{num_fold}/history.pt'
    else:
        path = f'{hp.path_model}/history.pt'
    history = torch.load(path)

    loss_keys = history['loss_keys']
    rate_keys = history['rate_keys']
    if 'Loss_Final' in loss_keys:
        loss_keys.remove('Loss_Final')

    loss = history['loss'][1:]
    rate = history['rate'][1:]
    k = len(loss[:, 1, 0][loss[:, 1, 0] > 0])
    loss = loss[:k, :, :]
    rate = rate[:k, :, :]

    if loss_titles is None: loss_titles = loss_keys
    if rate_titles is None: rate_titles = rate_keys

    N = len(loss_titles)
    n, m = (N // 3 + 1), 3

    plt.figure(figsize=(14, 4 * n))
    for i in range(N):
        plt.subplot(n, m, i + 1)
        plt.plot(loss[:, 0, i], label='Train', color='b', alpha=0.7)
        plt.plot(loss[:, 1, i], label='Valid', color='r', alpha=0.7)
        plt.ylim(get_ylim(loss[:, :, i], loss_titles[i], alpha=1.5))
        plt.title(loss_titles[i])
        plt.legend()

        if i == 2:
            plt.legend()
    plt.tight_layout()

    M = len(rate_titles)
    n2 = (M // 3 + 1)
    plt.figure(figsize=(14, 4 * n2))
    for j in range(M):
        plt.subplot(n2, m, j + 1)
        plt.plot(rate[:, 0, j], color='b', alpha=0.7)
        plt.plot(rate[:, 1, j], color='r', alpha=0.7)
        plt.ylim(get_ylim(rate[:, :, j], rate_titles[j], alpha=1.5))
        plt.title(rate_titles[j])
    plt.tight_layout()
    plt.show()

def tta_predict(model, x, pe=None, position_embedding=False, use_tta=True):
    if not use_tta:
        if position_embedding and pe is not None:
            pred, _, _ = model(x, pe)
        else:
            pred, _, _ = model(x)
        return torch.sigmoid(pred)
    
    # 4-way TTA
    preds = []
    flips = [(0,0), (1,0), (0,1), (1,1)]
    
    for flip_h, flip_w in flips:
        img = x.clone()
        if flip_h: img = torch.flip(img, dims=[2])
        if flip_w: img = torch.flip(img, dims=[3])
        
        with torch.no_grad():
            if position_embedding and pe is not None:
                pe_tta = pe.clone()
                if flip_h: pe_tta = torch.flip(pe_tta, dims=[2])
                if flip_w: pe_tta = torch.flip(pe_tta, dims=[3])
                pred, _, _ = model(img, pe_tta)
            else:
                pred, _, _ = model(img)
            
            pred = torch.sigmoid(pred)
            if flip_h: pred = torch.flip(pred, dims=[2])
            if flip_w: pred = torch.flip(pred, dims=[3])
        
        preds.append(pred)
    
    return torch.mean(torch.stack(preds), dim=0)

def inference_2(hp, test_loader, epoch, batch_size=None, n=0, clear=True, alpha=0.25, visualize=False, num_fold=None):
    print(f'Inference & Evaluation: {hp.name} / epoch={epoch} / fold={num_fold}')
    model = load_model(hp, epoch, num_fold).to(hp.device).eval()

    save_prefix = f'fold_{num_fold}'
    path_inf = os.path.join(hp.path_inference, str(epoch))
    path_eval = os.path.join(hp.path_evaluation, str(epoch))
    make_dir(path_inf)
    make_dir(path_eval)

    p_data, y_data, X_data, meta_data = [], [], [], []
    with torch.inference_mode():
        for data in test_loader:
            X, y = data[0].to(hp.device), data[1].to(hp.device)

            if hp.position_embedding:
                FOV_batch = []
                for fov_percent in data[3]:
                    tmp = []
                    for fov in fov_percent:
                        tmp.append(fov['FOV_Z_percent'])
                    FOV_batch.append(tmp)
                FOV_batch = torch.tensor(FOV_batch)
                pe = FOV_batch.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, X.shape[-2], X.shape[-1]).contiguous().to(hp.device)

            # p = model(X, pe) if hp.position_embedding else model(X)
            # p = torch.sigmoid(p)
            if hp.position_embedding:
                p_main = model(X, pe)
            else:
                p_main = model(X)
            p = torch.sigmoid(p_main)  # p_main만 사용!
            # p = tta_predict(model, X, pe if hp.position_embedding else None, hp.position_embedding)


            p_data.append(p.detach().float().cpu())
            y_data.append(y.cpu())
            X_data.append(X.cpu().clone())

            # meta_data.append(meta)

        p_data = torch.cat(p_data, dim=0)
        y_data = torch.cat(y_data, dim=0)
        X_data = torch.cat(X_data, dim=0)
        # meta_data = np.concatenate(meta_data, axis=0)

    with h5py.File(f'{path_inf}/{save_prefix}.h5', 'w') as f:
        f.create_dataset('p_data', data=p_data.numpy())
        f.create_dataset('y_data', data=y_data.numpy())
        # f.create_dataset('meta_data', data=meta_data)

    # Thresholding
    p_bin = (p_data > alpha).float()

    # Morphological closing post-processing
    kernel = np.ones((3, 3), np.uint8)  # 커널 크기 조정 가능
    p_bin_np = p_bin.squeeze(1).numpy().astype(np.uint8)  # (batch, H, W)
    p_bin_closed = []
    for i in range(p_bin_np.shape[0]):
        mask = p_bin_np[i]
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        p_bin_closed.append(closed)
    p_bin_closed = np.stack(p_bin_closed, axis=0)
    p_bin_closed = torch.from_numpy(p_bin_closed).unsqueeze(1).float()

    # Metrics 계산 (closing 적용된 결과 사용)
    metrics = {
        'ACC': ACC()(p_bin_closed, y_data).item(),
        'Dice': DiceScore()(p_bin_closed, y_data).item(),
        'Precision': Precision()(p_bin_closed, y_data).item(),
        'Recall': Recall()(p_bin_closed, y_data).item(),
        'Specificity': Specificity()(p_bin_closed, y_data).item()
    }

    torch.save(metrics, f'{path_eval}/{save_prefix}.pt')

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    if visualize:
        n_vis = p_data.shape[0] if n == 0 else min(n, p_data.shape[0])
        for i in range(n_vis):
            img = X_data[i][X_data.shape[1] // 2].numpy()
            mask = y_data[i].squeeze().numpy()
            pred = p_bin_closed[i].squeeze().numpy()  # closing된 결과 시각화
            prob = p_data[i].squeeze().numpy()

            plt.figure(figsize=(12, 3))
            plt.subplot(1, 4, 1); plt.imshow(img, cmap='gray'); plt.title(f'Input [{i}]'); plt.axis('off')
            plt.subplot(1, 4, 2); plt.imshow(mask, cmap='gray'); plt.title('GT'); plt.axis('off')
            plt.subplot(1, 4, 3); plt.imshow(pred, cmap='gray'); plt.title('Pred (Closed)'); plt.axis('off')
            plt.subplot(1, 4, 4); im = plt.imshow(prob, cmap='viridis'); plt.title('Prob')
            plt.colorbar(im); plt.axis('off')
            plt.tight_layout()
            plt.show()

    if clear:
        clear_output()

def inference(hp, test_loader, epoch, batch_size=None, n=0, clear=True, alpha=0.25, visualize=False, num_fold=None):
    print(f'Inference & Evaluation: {hp.name} / epoch={epoch} / fold={num_fold}')
    model = load_model(hp, epoch, num_fold).to(hp.device).eval()

    save_prefix = f'fold_{num_fold}'
    path_inf = os.path.join(hp.path_inference, str(epoch))
    path_eval = os.path.join(hp.path_evaluation, str(epoch))
    make_dir(path_inf)
    make_dir(path_eval)

    p_data, y_data, X_data, meta_data = [], [], [], []
    with torch.inference_mode():
        for data in test_loader:
            X, y = data[0].to(hp.device), data[1].to(hp.device)

            if hp.position_embedding:
                FOV_batch = []
                for fov_percent in data[3]:
                    tmp = []
                    for fov in fov_percent:
                        tmp.append(fov['FOV_Z_percent'])
                    FOV_batch.append(tmp)
                FOV_batch = torch.tensor(FOV_batch)
                pe = FOV_batch.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, X.shape[-2], X.shape[-1]).contiguous().to(hp.device)

            # p = model(X, pe) if hp.position_embedding else model(X)
            # p = torch.sigmoid(p)
            if hp.position_embedding:
                p_main, p_coarse, p_seg4 = model(X, pe)
            else:
                p_main, p_coarse, p_seg4 = model(X)
            p = torch.sigmoid(p_main)  # p_main만 사용!
            # p = tta_predict(model, X, pe if hp.position_embedding else None, hp.position_embedding)


            p_data.append(p.detach().float().cpu())
            y_data.append(y.cpu())
            X_data.append(X.cpu().clone())

            # meta_data.append(meta)

        p_data = torch.cat(p_data, dim=0)
        y_data = torch.cat(y_data, dim=0)
        X_data = torch.cat(X_data, dim=0)
        # meta_data = np.concatenate(meta_data, axis=0)

    with h5py.File(f'{path_inf}/{save_prefix}.h5', 'w') as f:
        f.create_dataset('p_data', data=p_data.numpy())
        f.create_dataset('y_data', data=y_data.numpy())
        # f.create_dataset('meta_data', data=meta_data)

    # Thresholding
    p_bin = (p_data > alpha).float()

    # Morphological closing post-processing
    kernel = np.ones((3, 3), np.uint8)  # 커널 크기 조정 가능
    p_bin_np = p_bin.squeeze(1).numpy().astype(np.uint8)  # (batch, H, W)
    p_bin_closed = []
    for i in range(p_bin_np.shape[0]):
        mask = p_bin_np[i]
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        p_bin_closed.append(closed)
    p_bin_closed = np.stack(p_bin_closed, axis=0)
    p_bin_closed = torch.from_numpy(p_bin_closed).unsqueeze(1).float()

    # Metrics 계산 (closing 적용된 결과 사용)
    metrics = {
        'ACC': ACC()(p_bin_closed, y_data).item(),
        'Dice': DiceScore()(p_bin_closed, y_data).item(),
        'Precision': Precision()(p_bin_closed, y_data).item(),
        'Recall': Recall()(p_bin_closed, y_data).item(),
        'Specificity': Specificity()(p_bin_closed, y_data).item()
    }

    torch.save(metrics, f'{path_eval}/{save_prefix}.pt')

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    if visualize:
        n_vis = p_data.shape[0] if n == 0 else min(n, p_data.shape[0])
        for i in range(n_vis):
            img = X_data[i][X_data.shape[1] // 2].numpy()
            mask = y_data[i].squeeze().numpy()
            pred = p_bin_closed[i].squeeze().numpy()  # closing된 결과 시각화
            prob = p_data[i].squeeze().numpy()

            plt.figure(figsize=(12, 3))
            plt.subplot(1, 4, 1); plt.imshow(img, cmap='gray'); plt.title(f'Input [{i}]'); plt.axis('off')
            plt.subplot(1, 4, 2); plt.imshow(mask, cmap='gray'); plt.title('GT'); plt.axis('off')
            plt.subplot(1, 4, 3); plt.imshow(pred, cmap='gray'); plt.title('Pred (Closed)'); plt.axis('off')
            plt.subplot(1, 4, 4); im = plt.imshow(prob, cmap='viridis'); plt.title('Prob')
            plt.colorbar(im); plt.axis('off')
            plt.tight_layout()
            plt.show()

    if clear:
        clear_output()

def print_total(list_model, list_epoch=None, metrics=['ACC'], col_width=20):
    n = 0

    for model in list_model:
        if model == '--':
            print('-' * (40 + col_width * len(metrics)))
            continue

        if list_epoch is None:
            list_epoch = listdir(f'../res/{model}/evaluation')

        for epoch in list_epoch:
            n += 1
            metric_results = {m: [] for m in metrics}

            eval_dir = f'../res/{model}/evaluation/{epoch}'
            eval_files = sorted([f for f in listdir(eval_dir) if f.endswith('.pt')])

            for eval_file in eval_files:
                res = torch.load(f'{eval_dir}/{eval_file}')
                for m in metrics:
                    metric_results[m].append(res[m])

            if n == 1:
                header = f'[{"epoch":>5}] {"model":<30}' + ''.join([f'{m:>{col_width}}' for m in metrics])
                print(header)

            result_line = f'[{epoch:>5}] {model:<30}'
            for m in metrics:
                values = torch.tensor(metric_results[m])
                mean = values.mean().item()
                std = values.std().item()
                formatted = f'{mean:.4f}±{std:.2f}'
                result_line += f'{formatted:>{col_width}}'
            print(result_line)

def visualize_dataloader_batch(data_loader, batch_index=0, sample_index=0):
    dataset = data_loader.dataset

    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset

    else:
        base_dataset = dataset

    context = None
    if isinstance(base_dataset, ConcatDataset):
        context = getattr(base_dataset.datasets[0], 'context', None)
    else:
        context = getattr(base_dataset, 'context', None)

    for i, (inputs, labels, metas, sideinfo) in enumerate(data_loader):
        if i == batch_index:
            input_tensor = inputs[sample_index]
            label_tensor = labels[sample_index]
            meta_list = metas[sample_index]
            sideinfo_list = sideinfo[sample_index]
            break
    else:
        print(f"batch_index {batch_index}가 DataLoader 범위를 벗어났습니다.")
        return

    if not isinstance(meta_list, list):
        print("meta_list가 리스트가 아닙니다.")
        print(f"meta_list: {meta_list}")
        return
    if context is not None and len(meta_list) <= context:
        print(f"meta_list 길이({len(meta_list)}) < context({context})")
        return

    input_np = input_tensor.cpu().numpy()
    label_np = label_tensor.squeeze(0).cpu().numpy()

    total_channels = input_np.shape[0]
    plt.figure(figsize=(4 * (total_channels + 1), 4))
    for i in range(total_channels):
        if i < len(meta_list):
            print(f"meta[{i}]: {meta_list[i]}")
            print(f"sideinfo[{i}]:")
            for key, value in sideinfo_list[i].items():
                print(f"  {key}: {value}")
        ax = plt.subplot(1, total_channels + 1, i + 1)
        ax.imshow(input_np[i], cmap='gray')
        ax.axis('off')

    ax = plt.subplot(1, total_channels + 1, total_channels + 1)
    ax.imshow(label_np, cmap='Reds', alpha=0.7)
    ax.axis('off')

    plt.tight_layout()
    plt.show()

def custom_collate_fn(batch):
    images, labels, metas, sideinfos = zip(*batch)  

    images = torch.stack(images)   
    labels = torch.stack(labels)   
    metas = [list(m) for m in metas]  
    sideinfos_out = [list(s) for s in sideinfos]

    return images, labels, metas, sideinfos_out

def load_h5_data(h5_path):
    """h5 데이터 로드"""
    try:
        with h5py.File(h5_path, 'r') as f:
            p_raw = f['p_data'][:]
            y_raw = f['y_data'][:]
        
        if p_raw.dtype == np.uint8:
            p_data = torch.tensor(p_raw.astype(np.float32) / 255.0)
        else:
            p_data = torch.tensor(p_raw).float()
        
        if y_raw.dtype == np.uint8:
            y_data = torch.tensor(y_raw.astype(np.float32) / 255.0)
        else:
            y_data = torch.tensor(y_raw).float()
        
        return p_data, y_data
    
    except Exception as e:
        raise ValueError(f"h5 로드 실패: {e}")


def evaluate_original_style_lesion(h5_path, alpha=0.05, iou_thresh=0.5, min_size=1):
    """
    폴드 하나(h5 파일)에 대해 병변 단위 TP/FP/FN 계산 후
    Dice, Precision, Recall, Specificity 리턴
    """
    p_data, y_data = load_h5_data(h5_path)

    p_bin = (p_data > alpha).float()
    y_bin = (y_data > 0.5).float()

    tp_total = fn_total = fp_total = 0

    for i in range(p_data.shape[0]):
        gt_labels, gt_num = ndimage.label(y_bin[i, 0].numpy())
        gt_sizes = ndimage.sum(y_bin[i, 0].numpy(), gt_labels, range(1, gt_num + 1))
        gt_valid = [idx + 1 for idx, size in enumerate(gt_sizes) if size >= min_size]

        pred_labels, pred_num = ndimage.label(p_bin[i, 0].numpy())
        pred_sizes = ndimage.sum(p_bin[i, 0].numpy(), pred_labels, range(1, pred_num + 1))
        pred_valid = [idx + 1 for idx, size in enumerate(pred_sizes) if size >= min_size]

        matched_gt = 0
        used_pred_labels = set()

        for g_label in gt_valid:
            g_mask = (gt_labels == g_label)
            gt_area = g_mask.sum()

            best_coverage = 0
            best_p = None

            for p_label in pred_valid:
                if p_label in used_pred_labels:
                    continue

                p_mask = (pred_labels == p_label)
                covered_gt = np.logical_and(g_mask, p_mask).sum()
                coverage = covered_gt / gt_area if gt_area > 0 else 0

                if coverage > best_coverage:
                    best_coverage = coverage
                    best_p = p_label

            if best_coverage >= iou_thresh and best_p is not None:
                matched_gt += 1
                used_pred_labels.add(best_p)

        tp_total += matched_gt
        fn_total += len(gt_valid) - matched_gt
        fp_total += len(pred_valid) - len(used_pred_labels)

    # 메트릭 계산
    dice = 2 * tp_total / max(2 * tp_total + fp_total + fn_total, 1)
    prec = tp_total / max(tp_total + fp_total, 1)
    rec = tp_total / max(tp_total + fn_total, 1)
    spec = 1 - fp_total / max(fp_total + (tp_total + fn_total), 1)

    # pixel-level ACC
    acc = ((p_bin == y_bin).float().mean()).item()

    return acc, dice, prec, rec, spec


def summarize_models(models, base_path="../res/",
                     alpha=0.05, iou_thresh=0.5, min_size=1, epochs=['best', '200']):
    """
    각 모델 × 에폭별 평균±표준편차 한 줄씩 출력 (best + 200epoch 모두)
    """
    print("=" * 140)
    print(f"{'Model':<32} {'Epoch':<8} {'ACC':<16} {'Dice':<16} {'Precision':<16} {'Recall':<16} {'Specificity':<16}")
    print("-" * 140)

    for model in models:
        model_name = os.path.basename(model)
        first_row = True

        for epoch in epochs:
            h5_pattern = os.path.join(base_path, model, "inference", str(epoch), "*.h5")
            h5_files = sorted(glob(h5_pattern))

            if not h5_files:
                label = model_name if first_row else ""
                print(f"{label:<32} {str(epoch):<8} (no h5 files)")
                first_row = False
                continue

            acc_list, dice_list, prec_list, rec_list, spec_list = [], [], [], [], []

            for h5_path in h5_files:
                try:
                    a, d, p, r, s = evaluate_original_style_lesion(
                        h5_path, alpha=alpha, iou_thresh=iou_thresh, min_size=min_size
                    )
                    acc_list.append(a)
                    dice_list.append(d)
                    prec_list.append(p)
                    rec_list.append(r)
                    spec_list.append(s)
                except:
                    pass

            label = model_name if first_row else ""
            if dice_list:
                acc_str  = f"{np.mean(acc_list):.4f} ± {np.std(acc_list):.2f}"
                dice_str = f"{np.mean(dice_list):.4f} ± {np.std(dice_list):.2f}"
                prec_str = f"{np.mean(prec_list):.4f} ± {np.std(prec_list):.2f}"
                rec_str  = f"{np.mean(rec_list):.4f} ± {np.std(rec_list):.2f}"
                spec_str = f"{np.mean(spec_list):.4f} ± {np.std(spec_list):.2f}"
                print(f"{label:<32} {str(epoch):<8} {acc_str:<16} {dice_str:<16} {prec_str:<16} {rec_str:<16} {spec_str}")
            else:
                print(f"{label:<32} {str(epoch):<8} (no valid results)")
            first_row = False

    print("=" * 140)