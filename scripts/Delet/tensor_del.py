import torch

def load_tensor_from_pt(file_path):
    """从pt文件加载数据并提取representations中的tensor"""
    try:
        # 加载pt文件
        data = torch.load(file_path)
    
        # 检查数据结构是否符合预期
        if 'representations' not in data:
            raise KeyError("数据中不包含'representations'键")
        
        # 提取键为33的tensor（根据你的数据结构）
        tensor_data = data['representations'] #.get(33)
        if tensor_data is None:
            raise KeyError("representations中不包含键为33的tensor")
            
        return tensor_data
    except Exception as e:
        print(f"加载数据出错: {e}")
        return None
    
def load_name(file_path):
    data = torch.load(file_path)

    name = data['label']

    return name


def subtract_tensors(file1_path, file2_path, output_path=None):
    """
    对两个pt文件中的tensor进行相减操作
    Args:
        file1_path: 第一个pt文件路径
        file2_path: 第二个pt文件路径
        output_path: 结果保存路径（可选）
    Returns:
        相减后的tensor
    """
    # 加载两个tensor
    tensor1 = load_tensor_from_pt(file1_path)
    tensor2 = load_tensor_from_pt(file2_path)
    
    name = load_name(file1_path)

    if tensor1 is None or tensor2 is None:
        return None
    
    # 检查tensor形状是否匹配
    if tensor1.shape != tensor2.shape:
        print(f"tensor形状不匹配: {tensor1.shape} vs {tensor2.shape}")
        return None
    
    # 执行相减操作
    result_tensor = tensor1 - tensor2
    print(f"相减完成，结果形状: {result_tensor.shape}")
    
    # 如果指定了输出路径，保存结果
    if output_path:
        try:
            # 保存为新的pt文件，保持类似的数据结构
            result_data = {
                'label': f"{name}",
                'representations': {33: result_tensor}
            }                    
            torch.save(result_data, output_path)
            print(f"结果已保存至: {output_path}")
        except Exception as e:
            print(f"保存结果出错: {e}")
    
    return result_tensor



# 使用示例
#if __name__ == "__main__":
    # 替换为你的文件路径
    #file1 = "newdata/esm2_650m/embeddings/mut66/rcsb_1A0N_B_I121L_7_25.pt"
    #file2 = "newdata/esm2_650m/embeddings/wt66/rcsb_1A0N_B_I121L_7_25.pt"
    #output = "newdata/Delet/result.pt"
    
    # 执行相减操作
    #result = subtract_tensors(file1, file2, output)
