import os
import h5py
import numpy as np
from PIL import Image
from utils.utils import expand_indices_grouped
from utils.utils import make_dir

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
from torchvision import models, transforms
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Dicom2dDataset(Dataset):
    def __init__(self, root_dir, target_dir, context=0, transform=None, only_lesion=False):
        self.context = context
        self.transform = transform
        self.only_lesion = only_lesion

        self.dcm, self.label, self.meta, self.sideinfo = [], [], [], []
        
        for subdir, _, files in os.walk(root_dir):
            folder_name = os.path.basename(subdir)
            if folder_name in target_dir:
                for file in files:
                    dcm_tmp, label_tmp, meta_tmp, sideinfo = [], [], [], {}
                    lesion = []
                    if file.endswith('.h5'):
                        path = os.path.join(subdir, file)
                        with h5py.File(path, 'r') as f:
                            dcm_tmp.append(f['dcm'][:])
                            label_tmp.append(f['label'][:])
                            meta_tmp.append(f['meta'][:])
                            for key in f.keys():
                                if key not in ['dcm', 'label', 'meta']:
                                    sideinfo[key] = f[key][:]
                        
                        dcm_tmp = np.concatenate(dcm_tmp, axis=0)
                        label_tmp = np.concatenate(label_tmp, axis=0)
                        meta_tmp = np.concatenate(meta_tmp, axis=0)

                        if only_lesion:
                            valid_indices = np.where(np.sum(label_tmp, axis=(1,2)).astype(bool) != 0)[0]
                        else:
                            valid_indices = np.arange(len(dcm_tmp))

                        indices= expand_indices_grouped(valid_indices, self.context, dcm_tmp.shape[0])

                        for indice in indices:
                            res = []
                            for i in indice:
                                if i is None:
                                    res.append(np.zeros_like(dcm_tmp[0])) 
                                else:
                                    res.append(dcm_tmp[i])  
                            
                            center_idx = indice[self.context]
                            if center_idx is None:
                                print('center is None')
                                continue
    
                            metas = [meta_tmp[i].decode('utf-8') if i is not None else 'padding' for i in indice] 
                            sideinfo_group = []
                            for i in indice:
                                side = {}
                                for k in sideinfo:
                                    v = sideinfo[k]
                                    if i is None:  # 패딩
                                        if np.isscalar(v[0]) or v.shape[1:] == ():
                                            side[k] = -0.001
                                        else:
                                            side[k] = np.zeros_like(v)
                                    else:
                                        val = v[i]
                                        # float 변환은 length-1일 때만!
                                        if np.isscalar(val) or np.array(val).shape == ():
                                            side[k] = float(val)
                                        else:
                                            side[k] = val  # 다차원은 그대로! (ndarray, list 등)
                                sideinfo_group.append(side)
                            self.dcm.append(res)
                            self.label.append(label_tmp[center_idx])
                            self.meta.append(metas)
                            self.sideinfo.append(sideinfo_group)


        n_total = len(self.dcm)
        n_lesion = 0

        for metas in self.meta:
            center_meta = metas[self.context]
            if center_meta != 'padding':  
                i = self.meta.index(metas)  # 현재 sample의 index
                center_label = self.label[i]
                if np.any(center_label > 0):
                    n_lesion += 1


        n_normal = n_total - n_lesion

        print(f"병변 슬라이스 수: {n_lesion}")
        print(f"정상 슬라이스 수: {n_normal}")
        print(f"전체 슬라이스 수: {n_total}")
        print(f"병변 비율: {n_lesion / n_total:.2%}")
        print("-" * 60)

        positive_pixels = sum(np.sum(lbl > 0) for lbl in self.label)
        total_pixels = sum(lbl.size for lbl in self.label)

        print(f"전체 픽셀 수: {total_pixels:,}")
        print(f"병변 픽셀 수: {positive_pixels:,}")
        print(f"병변 픽셀 비율: {positive_pixels / total_pixels:.4%}")
        print("-" * 60)

        print(f"dcm 개수: {len(self.dcm)}, label 개수: {len(self.label)}, meta 개수: {len(self.meta)}")


    def __len__(self):
        return len(self.dcm)

    def __getitem__(self, index):
        dcm_group = self.dcm[index]            # List[np.ndarray]
        label_2d = self.label[index]           # 2D np.ndarray
        meta_group = self.meta[index]          # List[str]
        sideinfo = self.sideinfo[index]        # List[float]

        input_np = np.stack([d.astype(np.float32) for d in dcm_group], axis=0)  # shape: (C, H, W)
        input_tensor = torch.from_numpy(input_np)
        label_tensor = torch.from_numpy(label_2d.astype(np.uint8)).unsqueeze(0).float()
        
        if self.transform:
            input_tensor = self.transform(input_tensor)  # transform이 numpy 기반인 경우
        return input_tensor, label_tensor, meta_group, sideinfo

    

class NormalSliceSelector:
    def __init__(self, dataset, seed, device=None, img_size=(192,192), n_clusters=200, context=None):
        self.dataset = dataset
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.n_clusters = n_clusters
        self.context = context if context is not None else getattr(dataset, 'context', 0)  # default 대표 context
        self.seed = seed

        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Identity()
        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)

        self.img_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    def get_lesion_bvalue(self):
        # 병변이 표시된 슬라이스들의 b_value(설정)를 확인해서, 노말 선택 시 맞춰야 할 기준값을 정한다.
        # (예: DWI에서 병변 마스크가 b=1000 슬라이스에만 있고 b=0 슬라이스엔 전혀 없는 경우,
        #  b=0에서 뽑은 '정상' 슬라이스는 병변 슬라이스와 다른 도메인이라 학습 일관성이 깨짐)
        bvals = []
        for i, lbl in enumerate(self.dataset.label):
            if np.any(lbl > 0):
                side = self.dataset.sideinfo[i][self.context]
                if 'b_value' in side:
                    bvals.append(side['b_value'])
        if not bvals:
            return None
        vals, counts = np.unique(bvals, return_counts=True)
        return vals[np.argmax(counts)]

    def get_normal_indices(self):
        # 정상 slice 인덱스만 추출 (병변 슬라이스와 같은 b_value 설정에서만 선택)
        target_bvalue = self.get_lesion_bvalue()
        indices = []
        for i, lbl in enumerate(self.dataset.label):
            if np.any(lbl > 0):
                continue
            if target_bvalue is not None:
                side = self.dataset.sideinfo[i][self.context]
                if side.get('b_value') != target_bvalue:
                    continue
            indices.append(i)
        return indices

    def get_feature(self, dcm_group):
        # dcm_group: [context 수, H, W] 또는 [H,W] (혹은 여러 context가 담긴 list/np.array)
        # 대표 context만 가져와서 resnet에 넣을 채널 형태로 변환
        if isinstance(dcm_group, (list, tuple)):
            img = dcm_group[self.context]
        elif isinstance(dcm_group, np.ndarray) and dcm_group.ndim == 3:
            img = dcm_group[self.context]
        else:
            img = dcm_group  # 단일 slice 케이스

        # img shape: [H,W] 또는 [H,W,C]
        # 1채널일 경우 → 3채널로 반복
        if img.ndim == 2:
            img_3ch = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        elif img.ndim == 3 and img.shape[2] == 1:
            img_3ch = np.repeat(img, 3, axis=2)
        elif img.ndim == 3 and img.shape == 3:
            img_3ch = img
        else:
            # 예외: 혹시 7채널 등 >3 일 때, 대표 context 이미지만 선택하면 이 코드는 실행 안됨. 혹시 shape가 이상하다면 디버깅 필요!
            raise ValueError(f"지원하지 않는 이미지 shape: {img.shape}")

        img_pil = Image.fromarray((img_3ch*255).clip(0,255).astype(np.uint8))
        img_tensor = self.img_transform(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.feature_extractor(img_tensor).cpu().numpy().squeeze()
        return feat

    def extract_features(self, indices):
        features = []
        metas = []
        sideinfos = []
        for idx in indices:
            dcm_group = self.dataset.dcm[idx]
            feat = self.get_feature(dcm_group)
            features.append(feat)
            metas.append(self.dataset.meta[idx][self.context])
            sideinfos.append(self.dataset.sideinfo[idx][self.context])
        features = np.array(features)
        return features, metas, sideinfos

    def cluster_and_select(self, features, indices):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        labels = kmeans.fit_predict(features)
        selected_cluster_indices = []
        for i in range(self.n_clusters):
            member_idx = np.where(labels == i)[0]
            center = kmeans.cluster_centers_[i]
            dists = np.linalg.norm(features[member_idx] - center, axis=1)
            sel = member_idx[np.argmin(dists)]
            selected_cluster_indices.append(sel)
        final_indices = [indices[i] for i in selected_cluster_indices]
        return final_indices, labels, kmeans.cluster_centers_

    def visualize_clusters(self, features, labels, selected_indices):
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(features)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', s=20, alpha=0.7)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('KMeans Clustering of Normal Slice Features')
        plt.colorbar(scatter, label='Cluster Label')
        if selected_indices is not None:
            plt.scatter(reduced[selected_indices, 0], reduced[selected_indices, 1],
                        c='red', edgecolors='black', s=80, marker='*', label='Selected Representative')
            plt.legend()
        plt.tight_layout()
        plt.show()

    def run_and_make_subset(self, visualize=True):
        print("Extracting normal slice indices...")
        normal_indices = self.get_normal_indices()
        print(f"Found {len(normal_indices)} normal slices.")
        print("Extracting features for normal slices...")
        features, metas, sideinfos = self.extract_features(normal_indices)
        print(f"Extracted features shape: {features.shape}")
        print(f"Running KMeans clustering with {self.n_clusters} clusters...")
        selected_indices, labels, centers = self.cluster_and_select(features, normal_indices)
        print(f"Selected {len(selected_indices)} representative normal slices.")
        if visualize:
            selected_in_features_idx = [normal_indices.index(idx) for idx in selected_indices]
            self.visualize_clusters(features, labels, selected_in_features_idx)

        selected_metas = [metas[normal_indices.index(idx)] for idx in selected_indices]
        selected_sideinfos = [sideinfos[normal_indices.index(idx)] for idx in selected_indices]

        normal_subset = Subset(self.dataset, selected_indices)
        return normal_subset, selected_indices, selected_metas, selected_sideinfos

import numpy as np
import torch
import matplotlib.pyplot as plt

class HF_LF_Visualizer:
    def __init__(self, dataset, radius=20):
        """
        :param dataset: (image, label_mask) 튜플로 구성된 데이터셋
        :param radius: FFT 필터 반경 (고주파/저주파 조절)
        """
        self.dataset = dataset
        self.radius = radius

    def low_pass_filter(self, image):
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)

        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        mask = np.zeros_like(image)
        y, x = np.ogrid[:rows, :cols]
        mask_area = (y - crow)**2 + (x - ccol)**2 <= self.radius**2
        mask[mask_area] = 1

        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return img_back

    def high_pass_filter(self, image):
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)

        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        mask = np.ones_like(image)
        y, x = np.ogrid[:rows, :cols]
        mask_area = (y - crow)**2 + (x - ccol)**2 <= self.radius**2
        mask[mask_area] = 0

        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return img_back

    def _get_image_and_label(self, sample):
        img_tensor = sample[0]
        label_tensor = sample[1] if len(sample) > 1 else None

        if isinstance(img_tensor, torch.Tensor):
            img = img_tensor.squeeze().cpu().numpy()
        else:
            img = np.array(img_tensor)

        if label_tensor is not None:
            if isinstance(label_tensor, torch.Tensor):
                label = label_tensor.squeeze().cpu().numpy()
            else:
                label = np.array(label_tensor)
        else:
            label = None
        
        return img, label

    def visualize_samples(self, num_samples=4):
        for i in range(num_samples):
            sample = self.dataset[i]
            img, label = self._get_image_and_label(sample)

            img_lf = self.low_pass_filter(img)
            img_hf = self.high_pass_filter(img)

            plt.figure(figsize=(16, 4))

            plt.subplot(1, 4, 1)
            plt.title('Original Image')
            plt.imshow(img, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 4, 2)
            plt.title('Label Mask')
            if label is not None:
                plt.imshow(label, cmap='gray')
            else:
                plt.text(0.5, 0.5, 'No Label', horizontalalignment='center', verticalalignment='center')
            plt.axis('off')

            plt.subplot(1, 4, 3)
            plt.title('Low Frequency')
            plt.imshow(img_lf, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 4, 4)
            plt.title('High Frequency')
            plt.imshow(img_hf, cmap='gray')
            plt.axis('off')

            plt.show()
