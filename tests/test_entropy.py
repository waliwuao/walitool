import numpy as np
import pytest
import walitool  # 这是你的 Rust 库

# --- 对照组：纯 Python 实现的熵权法 ---
def numpy_ewm(data):
    # 1. 计算列和
    col_sums = data.sum(axis=0)
    # 2. 计算比重 P (避免除以0，实际业务需预处理，这里假设数据合法)
    p_matrix = data / col_sums
    
    # 3. 计算熵值
    rows, cols = data.shape
    if rows <= 1:
        raise ValueError("Rows must be > 1")
        
    k = 1.0 / np.log(rows)
    
    # 计算 p * ln(p)，处理 p=0 的情况
    p_ln_p = np.zeros_like(p_matrix)
    non_zero = p_matrix > 0
    p_ln_p[non_zero] = p_matrix[non_zero] * np.log(p_matrix[non_zero])
    
    entropy = -k * p_ln_p.sum(axis=0)
    
    # 4. 计算差异系数
    divergence = 1.0 - entropy
    
    # 5. 归一化得到权重
    div_sum = divergence.sum()
    if div_sum == 0:
        return np.ones(cols) / cols
    return divergence / div_sum

# --- 测试用例 ---

def test_accuracy():
    """测试 Rust 算出的结果是否等于 NumPy 算出的结果"""
    # 构造一个 4行3列 的测试数据
    data = np.array([
        [10.0, 20.0, 30.0],
        [15.0, 25.0, 35.0],
        [12.0, 18.0, 28.0],
        [9.0,  21.0, 31.0]
    ])
    
    # Rust 结果
    rust_result = walitool.entropy_weight_method(data)
    # Python 对照组结果
    py_result = numpy_ewm(data)
    
    # 验证两者误差是否小于 1e-6
    np.testing.assert_allclose(rust_result, py_result, rtol=1e-6, atol=1e-6)
    print("\n✅ Accuracy Test Passed!")

def test_basic_logic():
    """测试基本逻辑：所有值相等时，权重应该相等"""
    data = np.ones((5, 3))
    rust_result = walitool.entropy_weight_method(data)
    expected = np.array([1/3, 1/3, 1/3])
    np.testing.assert_allclose(rust_result, expected, rtol=1e-6)
    print("✅ Logic Test Passed!")

def test_error_handling():
    """测试错误处理"""
    # 测试空数据
    with pytest.raises(ValueError):
        walitool.entropy_weight_method(np.array([]).reshape(0,0))
    print("✅ Error Handling Test Passed!")

if __name__ == "__main__":
    # 如果直接运行脚本
    test_accuracy()
    test_basic_logic()
    test_error_handling()