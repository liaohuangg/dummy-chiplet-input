
'''
generate_chiplet.py
该模块用于定义和处理芯粒（chiplet）对象及其相关操作。
主要内容：
- Chiplet类：用于表示芯粒的各项属性参数，包括名称、尺寸、类型、物理接口、工艺节点、功耗、中继功能、内部延迟、单元数量和电源凸点占比等。
- load_chiplets(json_path)：从指定的JSON文件路径加载芯粒信息，返回Chiplet对象列表。
- chiplets_to_dict(chiplet_list)：将Chiplet对象列表转换为字典格式，便于后续处理或输出。
适用场景：
本模块适用于芯粒参数的读取、封装和转换，便于芯粒相关的系统设计与分析。
'''
import json

class Chiplet:
    """
    Chiplet类用于表示一个芯粒（chiplet）的所有属性参数。

    属性说明：
    - name: 芯粒的名称（字符串）
    - dimensions: 芯粒的尺寸信息（字典，包含x和y轴长度）
    - type: 芯粒类型（如'compute'或'memory'等）
    - phys: 芯粒的物理接口位置列表，每个接口为一个字典
    - technology: 芯粒采用的工艺节点（字符串）
    - power: 芯粒的功耗（数值）
    - relay: 是否具备中继功能（布尔值）
    - internal_latency: 芯粒内部延迟（数值，单位为周期或时间）
    - unit_count: 芯粒内部单元数量（数值）
    - fraction_power_bumps: 电源凸点占比（数值或None）

    方法说明：
    - __init__: 初始化Chiplet对象并设置所有属性
    - __repr__: 返回Chiplet对象的字符串表示，便于调试和打印
    """
    def __init__(self, name, params):
        self.name = name
        # 芯粒在x和y轴上的尺寸，即芯粒的面积
        self.dimensions = params.get("dimensions", {})
        # 芯粒的类型
        self.type = params.get("type", "")
        # 芯粒的物理接口位置。
        # 匹配时，接口数量必须满足约束
        self.phys = params.get("phys", [])
        # 芯粒采用的工艺节点
        self.technology = params.get("technology", "")
        # 芯粒的功耗
        self.power = params.get("power", 0)
        # 芯粒是否具备中继功能，即是否可以作为信号或数据的转发节点
        self.relay = params.get("relay", False)
        # 芯粒的内部延迟，即信号处理的延迟周期数
        self.internal_latency = params.get("internal_latency", 0)
        # 芯粒内部单元数量（可忽略）
        self.unit_count = params.get("unit_count", 0)
        # 芯粒中电源凸点的占比（可忽略）
        self.fraction_power_bumps = params.get("fraction_power_bumps", None)

    def __repr__(self):
        return f"Chiplet(name={self.name}, type={self.type}, dimensions={self.dimensions}, phys={self.phys}, technology={self.technology}, power={self.power}, relay={self.relay}, internal_latency={self.internal_latency}, unit_count={self.unit_count}, fraction_power_bumps={self.fraction_power_bumps})"

def load_chiplets(json_path):
    with open(json_path, 'r') as f:
        chiplets_data = json.load(f)
    chiplet_objs = []
    for name, params in chiplets_data.items():
        chiplet = Chiplet(name, params)
        chiplet_objs.append(chiplet)
    return chiplet_objs

def chiplets_to_dict(chiplet_list):
    result = {}
    for chiplet in chiplet_list:
        result[chiplet.name] = {
            "dimensions": chiplet.dimensions,
            "type": chiplet.type,
            "phys": chiplet.phys,
            "technology": chiplet.technology,
            "power": chiplet.power,
            "relay": chiplet.relay,
            "internal_latency": chiplet.internal_latency,
            "unit_count": chiplet.unit_count,
            "fraction_power_bumps": chiplet.fraction_power_bumps
        }
    return result
    
if __name__ == "__main__":
    chiplet_json_path = './chiplet_input/chiplets.json'
    chiplet_list = load_chiplets(chiplet_json_path)
    chiplet_dict = chiplets_to_dict(chiplet_list)
    print(chiplet_dict)
