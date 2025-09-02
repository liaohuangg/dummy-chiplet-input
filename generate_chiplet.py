import json

class Chiplet:
    def __init__(self, name, params):
        self.name = name
        self.dimensions = params.get("dimensions", {})
        self.type = params.get("type", "")
        self.phys = params.get("phys", [])
        self.technology = params.get("technology", "")
        self.power = params.get("power", 0)
        self.relay = params.get("relay", False)
        self.internal_latency = params.get("internal_latency", 0)
        self.unit_count = params.get("unit_count", 0)
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
