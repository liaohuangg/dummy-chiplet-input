import json
import random
import os
from pathlib import Path

def create_directory_structure():
    """创建目录结构"""
    base_dir = Path("chiplet_library")
    subdirs = ["cpu", "gpu", "memory", "io", "noc"]
    
    base_dir.mkdir(exist_ok=True)
    for subdir in subdirs:
        (base_dir / subdir).mkdir(exist_ok=True)
    
    return base_dir

def generate_cpu_chiplet(chiplet_id):
    """生成CPU芯粒"""
    # CPU芯粒尺寸选项
    size_options = [
        {"x": 8, "y": 8},      # 标准CPU
        {"x": 10, "y": 10},    # 大型CPU
        {"x": 12, "y": 8},     # 矩形CPU
        {"x": 6, "y": 6},      # 小型CPU
        {"x": 14, "y": 10},    # 服务器CPU
    ]
    
    dimensions = random.choice(size_options)
    
    # CPU通常有较多接口用于连接缓存和内存
    phy_count = random.randint(4, 12)
    phys = generate_phys(dimensions, phy_count)
    
    # CPU芯粒特征
    core_count = random.choice([2, 4, 6, 8, 12, 16, 24, 32])
    
    return {
        "dimensions": dimensions,
        "type": "cpu",
        "phys": phys,
        "technology": random.choice(["7nm", "5nm", "3nm"]),
        "power": round(random.uniform(15, 150), 1),  # CPU功耗较高
        "relay": True,  # CPU通常支持中继
        "internal_latency": random.randint(1, 5),    # CPU延迟较低
        "unit_count": core_count,
        "specifications": {
            "core_count": core_count,
            "frequency_ghz": round(random.uniform(2.0, 5.0), 2),
            "cache_l3_mb": random.choice([8, 16, 32, 64, 128]),
            "architecture": random.choice(["x86", "ARM", "RISC-V"])
        }
    }

def generate_gpu_chiplet(chiplet_id):
    """生成GPU芯粒"""
    size_options = [
        {"x": 15, "y": 12},    # 标准GPU
        {"x": 20, "y": 15},    # 高端GPU
        {"x": 12, "y": 10},    # 中端GPU
        {"x": 8, "y": 8},      # 集成GPU
        {"x": 25, "y": 18},    # 数据中心GPU
    ]
    
    dimensions = random.choice(size_options)
    
    # GPU需要高带宽连接
    phy_count = random.randint(6, 16)
    phys = generate_phys(dimensions, phy_count)
    
    # GPU芯粒特征
    compute_units = random.choice([16, 32, 64, 128, 256, 512])
    
    return {
        "dimensions": dimensions,
        "type": "gpu",
        "phys": phys,
        "technology": random.choice(["7nm", "5nm", "4nm"]),
        "power": round(random.uniform(50, 400), 1),  # GPU功耗很高
        "relay": True,
        "internal_latency": random.randint(2, 8),
        "unit_count": compute_units,
        "specifications": {
            "compute_units": compute_units,
            "memory_bandwidth_gbps": random.randint(200, 1600),
            "tensor_performance_tops": round(random.uniform(50, 1000), 1),
            "architecture": random.choice(["RDNA", "Ada", "Ampere", "Hopper"])
        }
    }

def generate_memory_chiplet(chiplet_id):
    """生成内存芯粒"""
    memory_types = ["DDR5", "HBM3", "LPDDR5", "GDDR6", "Cache"]
    memory_type = random.choice(memory_types)
    
    # 根据内存类型调整尺寸
    if memory_type == "HBM3":
        size_options = [{"x": 5, "y": 5}, {"x": 6, "y": 6}]  # HBM较小
    elif memory_type == "Cache":
        size_options = [{"x": 4, "y": 4}, {"x": 6, "y": 4}]  # 缓存较小
    else:
        size_options = [{"x": 10, "y": 5}, {"x": 12, "y": 6}, {"x": 8, "y": 8}]
    
    dimensions = random.choice(size_options)
    
    # 内存芯粒接口相对较少
    phy_count = random.randint(2, 6)
    phys = generate_phys(dimensions, phy_count)
    
    # 根据内存类型设置参数
    if memory_type == "HBM3":
        capacity_gb = random.choice([8, 16, 24, 32])
        bandwidth = random.randint(600, 1200)
    elif memory_type == "DDR5":
        capacity_gb = random.choice([16, 32, 64, 128])
        bandwidth = random.randint(400, 800)
    elif memory_type == "Cache":
        capacity_gb = random.choice([0.032, 0.064, 0.128, 0.256])  # MB级别
        bandwidth = random.randint(1000, 2000)
    else:
        capacity_gb = random.choice([8, 16, 32])
        bandwidth = random.randint(400, 900)
    
    return {
        "dimensions": dimensions,
        "type": "memory",
        "phys": phys,
        "technology": random.choice(["14nm", "10nm", "7nm"]),
        "power": round(random.uniform(2, 20), 1),
        "relay": False,  # 内存通常不中继
        "internal_latency": random.randint(3, 15),
        "unit_count": random.randint(1, 8),
        "specifications": {
            "memory_type": memory_type,
            "capacity_gb": capacity_gb,
            "bandwidth_gbps": bandwidth,
            "channels": random.choice([1, 2, 4, 8])
        }
    }

def generate_io_chiplet(chiplet_id):
    """生成IO芯粒"""
    io_types = ["PCIe", "USB", "Network", "Storage", "Display"]
    io_type = random.choice(io_types)
    
    size_options = [
        {"x": 6, "y": 8},      # 标准IO
        {"x": 8, "y": 6},      # 宽IO
        {"x": 5, "y": 5},      # 小IO
        {"x": 10, "y": 8},     # 大IO
    ]
    
    dimensions = random.choice(size_options)
    
    # IO芯粒接口数量中等
    phy_count = random.randint(3, 8)
    phys = generate_phys(dimensions, phy_count)
    
    # 根据IO类型设置参数
    if io_type == "PCIe":
        lanes = random.choice([4, 8, 16])
        speed = random.choice(["Gen4", "Gen5", "Gen6"])
    elif io_type == "Network":
        speed = random.choice(["1G", "10G", "25G", "100G"])
        lanes = random.choice([1, 2, 4])
    else:
        lanes = random.choice([1, 2, 4])
        speed = "Standard"
    
    return {
        "dimensions": dimensions,
        "type": "io",
        "phys": phys,
        "technology": random.choice(["12nm", "10nm", "7nm"]),
        "power": round(random.uniform(3, 25), 1),
        "relay": random.choice([True, False]),
        "internal_latency": random.randint(2, 10),
        "unit_count": random.randint(1, 4),
        "specifications": {
            "io_type": io_type,
            "lanes": lanes,
            "speed": speed,
            "protocols": random.sample(["PCIe", "USB3", "SATA", "Ethernet"], random.randint(1, 3))
        }
    }

def generate_noc_chiplet(chiplet_id):
    """生成NoC(Network on Chip)芯粒"""
    size_options = [
        {"x": 4, "y": 4},      # 小型路由器
        {"x": 6, "y": 6},      # 标准路由器
        {"x": 8, "y": 4},      # 交换机
        {"x": 3, "y": 3},      # 微型路由器
        {"x": 10, "y": 6},     # 大型交换机
    ]
    
    dimensions = random.choice(size_options)
    
    # NoC芯粒通常有很多连接接口
    phy_count = random.randint(8, 20)
    phys = generate_phys(dimensions, phy_count)
    
    # NoC类型
    noc_types = ["Router", "Switch", "Bridge", "Gateway"]
    noc_type = random.choice(noc_types)
    
    return {
        "dimensions": dimensions,
        "type": "noc",
        "phys": phys,
        "technology": random.choice(["10nm", "7nm", "5nm"]),
        "power": round(random.uniform(1, 15), 1),  # NoC功耗较低
        "relay": True,  # NoC主要功能就是中继
        "internal_latency": random.randint(1, 3),  # NoC延迟极低
        "unit_count": random.randint(4, 32),       # 路由单元数
        "specifications": {
            "noc_type": noc_type,
            "ports": random.choice([4, 6, 8, 12, 16]),
            "bandwidth_per_port_gbps": random.randint(50, 400),
            "topology": random.choice(["Mesh", "Torus", "Ring", "Tree", "Crossbar"])
        }
    }

def generate_phys(dimensions, count):
    """生成物理接口位置"""
    phys = []
    x_max, y_max = dimensions["x"], dimensions["y"]
    
    # 计算每条边应该放置的接口数量
    edge_count = max(1, count // 4)
    remaining = count
    
    edges = ["top", "bottom", "left", "right"]
    random.shuffle(edges)
    
    for edge in edges:
        if remaining <= 0:
            break
            
        current_edge_count = min(edge_count, remaining)
        
        for i in range(current_edge_count):
            if edge == "top":
                x = round(random.uniform(0.5, x_max - 0.5), 2)
                y = round(y_max - 0.25, 2)
            elif edge == "bottom":
                x = round(random.uniform(0.5, x_max - 0.5), 2)
                y = 0.25
            elif edge == "left":
                x = 0.25
                y = round(random.uniform(0.5, y_max - 0.5), 2)
            else:  # right
                x = round(x_max - 0.25, 2)
                y = round(random.uniform(0.5, y_max - 0.5), 2)
            
            phy = {"x": x, "y": y}
            
            # 随机添加bump区域属性
            if random.random() < 0.25:
                phy["fraction_bump_area"] = round(random.uniform(0.1, 0.4), 2)
            
            phys.append(phy)
            remaining -= 1
    
    return phys

def generate_chiplet_library():
    """生成完整的芯粒库"""
    print("🚀 开始生成五种类型的芯粒库...")
    
    # 创建目录结构
    base_dir = create_directory_structure()
    
    # 定义每种类型的数量
    counts = {
        "cpu": 100,
        "gpu": 80,
        "memory": 150,
        "io": 120,
        "noc": 150
    }
    
    generators = {
        "cpu": generate_cpu_chiplet,
        "gpu": generate_gpu_chiplet,
        "memory": generate_memory_chiplet,
        "io": generate_io_chiplet,
        "noc": generate_noc_chiplet
    }
    
    total_generated = 0
    
    for chiplet_type, count in counts.items():
        print(f"📦 正在生成 {count} 个 {chiplet_type.upper()} 芯粒...")
        
        chiplets = {}
        generator = generators[chiplet_type]
        
        for i in range(1, count + 1):
            chiplet_name = f"{chiplet_type}_chiplet_{i:03d}"
            chiplets[chiplet_name] = generator(i)
        
        # 保存到对应文件夹
        output_file = base_dir / chiplet_type / f"{chiplet_type}_chiplets.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chiplets, f, indent=4, ensure_ascii=False)
        
        print(f"✅ {chiplet_type.upper()} 芯粒已保存到 {output_file}")
        total_generated += count
    
    # 生成汇总文件
    print("📋 生成汇总文件...")
    summary = {
        "library_info": {
            "total_chiplets": total_generated,
            "types": list(counts.keys()),
            "generation_date": "2025-09-08",
            "description": "Five-type chiplet library for matching algorithm research"
        },
        "type_counts": counts,
        "file_structure": {
            chiplet_type: f"{chiplet_type}/{chiplet_type}_chiplets.json" 
            for chiplet_type in counts.keys()
        }
    }
    
    summary_file = base_dir / "library_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    print(f"芯粒库生成完成！")
    print(f"总计生成: {total_generated} 个芯粒")
    print(f"文件结构:")
    print(f"   chiplet_library/")
    for chiplet_type, count in counts.items():
        print(f"   ├── {chiplet_type}/ ({count} 个芯粒)")
        print(f"   │   └── {chiplet_type}_chiplets.json")
    print(f"   └── library_summary.json")

if __name__ == "__main__":
    generate_chiplet_library()