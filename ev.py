import os
import torch
import numpy as np
from Bio.PDB import PDBParser
from model import ProteinTransformer  # 假设你已经有了这个模型

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
                if 'CA' in residue and residue.get_resname() in AA3_to_AA1:
                    ca = residue['CA'].get_coord()
                    coords.append(ca)
                    sequence.append(AA3_to_AA1[residue.get_resname()])
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


# 使用模型进行预测
def predict_sequence(model, coords, device):
    coords_tensor = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(device)  # (1, L, 3)

    # 创建一个二维的mask，形状为 (1, L)，表示所有位置都是有效的
    mask = torch.ones(coords_tensor.shape[1], dtype=torch.bool).to(device).unsqueeze(0)  # (1, L)

    with torch.no_grad():
        logits = model(coords_tensor, padding_mask=~mask)

    preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (L, )
    return preds


# 转换为氨基酸序列
def sequence_from_predictions(preds, idx_to_aa):
    return ''.join([idx_to_aa[idx] for idx in preds])


# 加载训练好的模型
def load_model(model_path, device):
    model = ProteinTransformer(max_len=2048).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# 主函数
def main():
    # 直接指定参数，而不是使用 argparse
    pdb_file = "1io6.pdb"  # 这里修改为你的 PDB 文件路径
    model_path = "checkpoints/model_epoch_108.pth"  # 这里修改为你的模型路径
    output_dir = "./"  # 输出目录
    max_length = 2048  # 最大序列长度

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 提取蛋白质的CA坐标和氨基酸序列
    coords, seq = extract_structure_sequence(pdb_file)

    # 如果没有提取到数据，打印错误信息并退出
    if coords is None or seq is None:
        print(f"Error: Unable to extract CA coordinates or sequence from {pdb_file}")
        return

    # 将氨基酸序列进行独热编码
    seq_one_hot = one_hot_encode(seq)

    # 加载训练好的模型
    model = load_model(model_path, device)

    # 假设你有一个映射：索引 -> 氨基酸字母（根据你的训练数据设置）
    idx_to_aa = {i: aa for i, aa in enumerate(AA_vocab)}  # 基于AA_vocab创建索引到氨基酸的映射

    # 使用模型进行预测
    preds = predict_sequence(model, coords, device)

    # 获取氨基酸序列
    sequence = sequence_from_predictions(preds, idx_to_aa)

    # 输出或保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{os.path.basename(pdb_file)}_predicted_sequence.txt')
    with open(output_file, 'w') as f:
        f.write(sequence)

    print(f"Predicted sequence saved to {output_file}")
    print(f"Predicted Sequence: {sequence}")


if __name__ == '__main__':
    main()
