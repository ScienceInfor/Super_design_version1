import os
import numpy as np
import torch
import pickle
from Bio.PDB import PDBParser
from tqdm import tqdm  # 导入进度条库

# 三字母到一字母的氨基酸映射
AA3_to_AA1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# 氨基酸字典
AA_vocab = "ACDEFGHIKLMNPQRSTVWY"
AA_to_idx = {aa: i for i, aa in enumerate(AA_vocab)}


# 独热编码函数
def one_hot_encode(seq):
    one_hot = np.zeros((len(seq), len(AA_vocab)), dtype=np.float32)
    for i, aa in enumerate(seq):
        if aa in AA_to_idx:
            one_hot[i, AA_to_idx[aa]] = 1.0
    return one_hot


# 提取 alpha carbon 坐标和氨基酸序列
def extract_structure_sequence(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    coords, sequence = [], []

    for model in structure:
        for chain in model:
            for residue in chain:
                # 确保是有效氨基酸，并且含有 alpha carbon (CA)
                if 'CA' in residue and residue.get_resname() in AA3_to_AA1:
                    ca = residue['CA'].get_coord()
                    coords.append(ca)
                    sequence.append(AA3_to_AA1[residue.get_resname()])
                # 如果是无效氨基酸（如X），跳过
                elif residue.get_resname() not in AA3_to_AA1:
                    continue
            break  # 只取第一个链
        break  # 只取第一个模型

    coords = np.array(coords)

    # 如果没有 CA 坐标，返回空数据
    if coords.shape[0] == 0:
        return None, None

    # 消除质心
    centroid = np.mean(coords, axis=0)
    coords -= centroid

    return coords, ''.join(sequence)


# 主预处理函数
def preprocess_and_save(pdb_dir, save_path='data.pkl'):
    dataset = []
    # 获取所有PDB文件列表
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    # 添加进度条
    progress_bar = tqdm(pdb_files, desc="Processing PDBs", unit="file")
    skipped_count = 0  # 记录跳过的文件数
    for filename in progress_bar:
        file_path = os.path.join(pdb_dir, filename)
        coords, seq = extract_structure_sequence(file_path)

        # 更新进度条描述显示当前文件名
        progress_bar.set_postfix(file=filename[:10], skipped=f"{skipped_count}")

        # 检查数据有效性
        if coords is None or seq is None:
            skipped_count += 1
            continue
        if len(seq) < 10:
            skipped_count += 1
            continue
        if len(seq) != len(coords):
            skipped_count += 1
            continue

        seq_one_hot = one_hot_encode(seq)
        dataset.append({
            'coords': coords.astype(np.float32),
            'seq': seq_one_hot  # 使用 one-hot 编码的序列
        })

    # 保存数据
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)

    # 打印总结信息
    print(f"\nProcessing complete! Saved {len(dataset)} proteins to {save_path}")
    print(f"Skipped {skipped_count} files due to invalid data")


if __name__ == '__main__':
    preprocess_and_save('5000pdbs')