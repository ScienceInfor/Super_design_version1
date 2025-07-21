import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle

class ProteinDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        # 过滤掉无效数据，确保数据是有效的
        self.data = [item for item in self.data if item['coords'].size != 0 and item['seq'].size != 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        coords = torch.tensor(item['coords'], dtype=torch.float32)  # (L, 3)
        seq = torch.tensor(item['seq'], dtype=torch.float32)        # (L, 20) one-hot
        return coords, seq

def collate_fn(batch):
    """
    自定义collate函数，对batch中的样本进行填充。
    """
    max_len = max([item[0].size(0) for item in batch])

    padded_coords = []
    padded_seqs = []

    for coords, seq in batch:
        pad_length = max_len - coords.size(0)

        # 填充 coords: (L, 3) → (max_len, 3)
        padded_coords.append(F.pad(coords, (0, 0, 0, pad_length), value=0))

        # 填充 seq: (L, 20) → (max_len, 20)
        padded_seqs.append(F.pad(seq, (0, 0, 0, pad_length), value=0))

    padded_coords = torch.stack(padded_coords, dim=0)  # (B, max_len, 3)
    padded_seqs = torch.stack(padded_seqs, dim=0)      # (B, max_len, 20)

    return padded_coords, padded_seqs
