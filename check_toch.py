import torch

def print_divider(title):
    print("\n" + "="*50)
    print(f" {title} ")
    print("="*50)

# 1. 基础信息检测
print_divider("1. PyTorch 基础信息")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

# 2. CUDA 可用性检测
print_divider("2. CUDA 可用性检测")
if torch.cuda.is_available():
    print("✅ CUDA 可用")
    print(f"当前设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
    print(f"设备数量: {torch.cuda.device_count()}")
    
    # CUDA功能测试
    print("\n🔧 CUDA 功能测试...")
    try:
        a = torch.tensor([1.0, 2.0, 3.0]).cuda()
        b = torch.tensor([4.0, 5.0, 6.0]).cuda()
        c = a + b
        print("✅ CUDA 张量计算成功")
        print(f"计算结果: {c.cpu().numpy()}")
    except Exception as e:
        print(f"❌ CUDA 测试失败: {str(e)}")
else:
    print("❌ CUDA 不可用")
    print("提示: 请检查是否正确安装CUDA PyTorch版本")

# 3. cuDNN 检测
print_divider("3. cuDNN 功能检测")
if torch.cuda.is_available():
    # 使用卷积操作测试cuDNN
    print("🔧 运行cuDNN卷积测试...")
    try:
        # 创建随机的输入和卷积核
        input = torch.randn(1, 3, 32, 32).cuda()
        conv = torch.nn.Conv2d(3, 6, 3).cuda()
        output = conv(input)
        
        print(f"✅ cuDNN 卷积运算成功")
        print(f"输出维度: {output.shape}")
        
        # 检测cuDNN版本
        if torch.backends.cudnn.enabled:
            print(f"cuDNN 版本: v{torch.backends.cudnn.version()}")
        else:
            print("⚠️ cuDNN 已安装但未启用")
            
    except Exception as e:
        print(f"❌ cuDNN 测试失败: {str(e)}")
        print("提示: 请检查cuDNN与CUDA版本的兼容性")
else:
    print("⏩ 跳过cuDNN测试 (CUDA不可用)")

# 4. 设备性能基准测试
print_divider("4. GPU 性能基准测试")
if torch.cuda.is_available():
    print("⏱️ 运行矩阵乘法基准测试...")
    try:
        device = torch.device("cuda")
        x = torch.rand(10000, 10000, device=device)
        y = torch.rand(10000, 10000, device=device)
        
        # 预热
        torch.cuda.synchronize()
        torch.matmul(x, y)
        torch.cuda.synchronize()
        
        # 正式测试
        import time
        start = time.time()
        torch.matmul(x, y)
        torch.cuda.synchronize()
        duration = time.time() - start
        
        print(f"✅ GPU 计算完成 (耗时: {duration:.4f}秒)")
        print(f"预计性能: {1e9/(duration):.0f} FLOPS")
    except Exception as e:
        print(f"❌ 性能测试失败: {str(e)}")
else:
    print("⏩ 跳过性能测试 (CUDA不可用)")

print_divider("测试完成")
print(f"最终状态: {'✅ 所有测试通过' if torch.cuda.is_available() else '❌ 存在未通过的测试'}")
