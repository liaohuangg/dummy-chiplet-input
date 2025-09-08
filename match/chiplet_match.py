import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class MatchingRequirement:
    """匹配需求定义"""
    chiplet_types: List[str]  # 需要的芯粒类型
    min_counts: Dict[str, int]  # 每种类型的最小数量
    max_counts: Dict[str, int]  # 每种类型的最大数量
    power_budget: float  # 功耗预算
    area_budget: float  # 面积预算
    performance_target: float  # 性能目标
    constraints: Dict  # 其他约束条件

@dataclass
class ChipletMatch:
    """芯粒匹配结果"""
    selected_chiplets: List[Dict]
    total_power: float
    total_area: float
    performance_score: float
    cost_score: float
    match_score: float

class ChipletMatcher:
    """芯粒匹配器"""
    
    def __init__(self, chiplet_library_path: str):
        """初始化匹配器"""
        self.chiplet_library = self.load_chiplet_library(chiplet_library_path)
        self.chiplets_by_type = self.organize_by_type()
        
    def load_chiplet_library(self, library_path: str) -> Dict:
        """加载芯粒库"""
        library = {}
        base_path = Path(library_path)
        
        # 加载各类型芯粒
        types = ["cpu", "gpu", "memory", "io", "noc"]
        for chiplet_type in types:
            file_path = base_path / chiplet_type / f"{chiplet_type}_chiplets.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    type_chiplets = json.load(f)
                    library.update(type_chiplets)
        
        print(f" 加载了 {len(library)} 个芯粒")
        return library

    def organize_by_type(self) -> Dict[str, List]:
        """按类型组织芯粒"""
        organized = {"cpu": [], "gpu": [], "memory": [], "io": [], "noc": []}
        

        
        for name, chiplet in self.chiplet_library.items():
            #name是chiplet的名字，chiplet是具体信息
            #例如：name=cpu_chiplet_001
            #chiplet={'type': 'cpu', 'unit_count': 4, 'power': 65, 'dimensions': {'x': 10, 'y': 10}, 'internal_latency': 10}
            chiplet_type = chiplet.get("type", "unknown")
            if chiplet_type in organized:
                chiplet["name"] = name
                organized[chiplet_type].append(chiplet)

    
        # print(organized["cpu"][1])
        return organized
    
    def calculate_chiplet_area(self, chiplet: Dict) -> float:
        """计算芯粒面积"""
        dims = chiplet.get("dimensions", {})
        return dims.get("x", 0) * dims.get("y", 0)
    
    def calculate_performance_score(self, chiplet: Dict) -> float:
        """计算性能分数"""
        chiplet_type = chiplet.get("type", "")
        unit_count = chiplet.get("unit_count", 1)
        latency = chiplet.get("internal_latency", 1)
        
        # 不同类型芯粒的性能计算方式不同
        if chiplet_type == "cpu":
            return unit_count * 100 / latency  # 核心数/延迟
        elif chiplet_type == "gpu":
            return unit_count * 50 / latency   # 计算单元数/延迟
        elif chiplet_type == "memory":
            return unit_count * 200 / latency  # 内存带宽相关
        elif chiplet_type == "io":
            return unit_count * 80 / latency   # IO吞吐量相关
        elif chiplet_type == "noc":
            return unit_count * 150 / latency  # 网络性能相关




     


    def random_matching(self, requirement: MatchingRequirement) -> ChipletMatch:
        """随机匹配算法"""
        selected = []
        total_power = 0
        total_area = 0
        total_performance = 0
        
        # 按需求匹配各类型芯粒
        for chiplet_type in requirement.chiplet_types:
            if chiplet_type not in self.chiplets_by_type:
                continue
                
            candidates = self.chiplets_by_type[chiplet_type]
            min_count = requirement.min_counts.get(chiplet_type, 1)
            max_count = requirement.max_counts.get(chiplet_type, len(candidates))
            
            count = np.random.randint(min_count, max_count+1)
            selected_chiplets = np.random.choice(candidates, size=min(count, len(candidates)), replace=False)
            
            for chiplet in selected_chiplets:
                power = chiplet.get("power", 1)
                area = self.calculate_chiplet_area(chiplet)
                perf = self.calculate_performance_score(chiplet)
                
                # 检查是否超出预算
                if (total_power + power <= requirement.power_budget and 
                    total_area + area <= requirement.area_budget):
                    selected.append(chiplet)
                    total_power += power
                    total_area += area
                    total_performance += perf
        
        # 计算匹配分数
        power_ratio = total_power / requirement.power_budget if requirement.power_budget > 0 else 0
        area_ratio = total_area / requirement.area_budget if requirement.area_budget > 0 else 0
        perf_ratio = total_performance / requirement.performance_target if requirement.performance_target > 0 else 1
        
        # 综合评分 (性能越高越好，功耗面积越低越好)
        match_score = perf_ratio * 0.5 - power_ratio * 0.3 - area_ratio * 0.2
        
        return ChipletMatch(
            selected_chiplets=selected,
            total_power=total_power,
            total_area=total_area,
            performance_score=total_performance,
            cost_score=power_ratio + area_ratio,
            match_score=match_score
        )

    
    def greedy_matching(self, requirement: MatchingRequirement) -> ChipletMatch:
        """贪心匹配算法"""
        selected = []
        total_power = 0
        total_area = 0
        total_performance = 0
        
        # 按需求匹配各类型芯粒
        for chiplet_type in requirement.chiplet_types:
            if chiplet_type not in self.chiplets_by_type:
                continue
                
            candidates = self.chiplets_by_type[chiplet_type]
            min_count = requirement.min_counts.get(chiplet_type, 1)
            max_count = requirement.max_counts.get(chiplet_type, len(candidates))
            
            # 按性能功耗比排序
            candidates_scored = []
            for chiplet in candidates:
                perf_score = self.calculate_performance_score(chiplet)
                power = chiplet.get("power", 1)
                area = self.calculate_chiplet_area(chiplet)
                
                # 检查是否超出预算
                if (total_power + power <= requirement.power_budget and 
                    total_area + area <= requirement.area_budget):
                    
                    efficiency = perf_score / (power + 0.1)  # 性能功耗比
                    candidates_scored.append((chiplet, perf_score, power, area, efficiency))
            
            # 按效率排序，选择最优的
            candidates_scored.sort(key=lambda x: x[4], reverse=True)
            
            count = 0
            for chiplet, perf, power, area, eff in candidates_scored:
                if count >= max_count:
                    break
                if count < min_count or (total_power + power <= requirement.power_budget and 
                                       total_area + area <= requirement.area_budget):
                    selected.append(chiplet)
                    total_power += power
                    total_area += area
                    total_performance += perf
                    count += 1
        
        # 计算匹配分数
        power_ratio = total_power / requirement.power_budget if requirement.power_budget > 0 else 0
        area_ratio = total_area / requirement.area_budget if requirement.area_budget > 0 else 0
        perf_ratio = total_performance / requirement.performance_target if requirement.performance_target > 0 else 1
        
        # 综合评分 (性能越高越好，功耗面积越低越好)
        match_score = perf_ratio * 0.5 - power_ratio * 0.3 - area_ratio * 0.2
        
        return ChipletMatch(
            selected_chiplets=selected,
            total_power=total_power,
            total_area=total_area,
            performance_score=total_performance,
            cost_score=power_ratio + area_ratio,
            match_score=match_score
        )
    
    def genetic_algorithm_matching(self, requirement: MatchingRequirement, 
                                 population_size: int = 50, generations: int = 100) -> ChipletMatch:
        """遗传算法匹配"""
        
        def create_random_solution():
            """创建随机解"""
            solution = []
            for chiplet_type in requirement.chiplet_types:
                if chiplet_type in self.chiplets_by_type:
                    candidates = self.chiplets_by_type[chiplet_type]
                    min_count = requirement.min_counts.get(chiplet_type, 1)
                    max_count = min(requirement.max_counts.get(chiplet_type, 5), len(candidates))
                    if max_count <= min_count :
                       max_count= min_count + 1
                    count = np.random.randint(min_count, max_count+1)
                    selected = np.random.choice(candidates, size=min(count, len(candidates)), replace=False)
                    solution.extend(selected)
            return solution
        
        def evaluate_solution(solution):
            """评估解的适应度"""
            total_power = sum(c.get("power", 0) for c in solution)
            total_area = sum(self.calculate_chiplet_area(c) for c in solution)
            total_perf = sum(self.calculate_performance_score(c) for c in solution)
            
            # 惩罚超出预算的解
            power_penalty = max(0, total_power - requirement.power_budget) * 10
            area_penalty = max(0, total_area - requirement.area_budget) * 10
            
            fitness = total_perf - power_penalty - area_penalty
            return fitness, total_power, total_area, total_perf
        
        def crossover(parent1, parent2):
            """交叉操作"""
            crossover_point = len(parent1) // 2
            child = parent1[:crossover_point] + parent2[crossover_point:]
            return child
        
        def mutate(solution, mutation_rate=0.1):
            """变异操作"""
            if np.random.random() < mutation_rate:
                # 随机替换一个芯粒
                if solution:
                    idx = np.random.randint(0, len(solution))
                    old_type = solution[idx].get("type", "")
                    if old_type in self.chiplets_by_type:
                        candidates = self.chiplets_by_type[old_type]
                        solution[idx] = np.random.choice(candidates)
            return solution
        
        # 初始化种群
        population = [create_random_solution() for _ in range(population_size)]
        
        best_solution = None
        best_fitness = float('-inf')
        
        for generation in range(generations):
            # 评估种群
            fitness_scores = []
            for solution in population:
                fitness, power, area, perf = evaluate_solution(solution)
                fitness_scores.append((fitness, solution, power, area, perf))
            
            # 找到最佳解
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            if fitness_scores[0][0] > best_fitness:
                best_fitness = fitness_scores[0][0]
                best_solution = fitness_scores[0]
            
            # 选择、交叉、变异
            new_population = []
            elite_size = population_size // 4
            
            # 保留精英
            for i in range(elite_size):
                new_population.append(fitness_scores[i][1])
            
            # 生成新个体
            while len(new_population) < population_size:
                parent1 = fitness_scores[np.random.randint(0, population_size//2)][1]
                parent2 = fitness_scores[np.random.randint(0, population_size//2)][1]
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)
            
            population = new_population
        
        # 返回最佳匹配结果
        _, solution, power, area, perf = best_solution
        
        # 使用与其他算法相同的评分标准
        power_ratio = power / requirement.power_budget if requirement.power_budget > 0 else 0
        area_ratio = area / requirement.area_budget if requirement.area_budget > 0 else 0
        perf_ratio = perf / requirement.performance_target if requirement.performance_target > 0 else 1
        
        # 综合评分 (性能越高越好，功耗面积越低越好)
        match_score = perf_ratio * 0.5 - power_ratio * 0.3 - area_ratio * 0.2
        
        return ChipletMatch(
            selected_chiplets=solution,
            total_power=power,
            total_area=area,
            performance_score=perf,
            cost_score=power_ratio + area_ratio,
            match_score=match_score
        )

def main():
    """主函数 - 演示芯粒匹配"""
    
    # 初始化匹配器
    matcher = ChipletMatcher("../chiplet_library")
    n=10
    
    # 定义匹配需求
    requirement = MatchingRequirement(
        chiplet_types=["cpu", "gpu", "memory", "io", "noc"],
        min_counts={"cpu": 10, "gpu": 10, "memory": 20, "io": 10, "noc": 20},
        max_counts={"cpu": 20, "gpu": 20, "memory": 40, "io": 50, "noc": 40},
        power_budget=1000.0,      # 500W功耗预算
        area_budget=1000.0,      # 1000平方单位面积预算
        performance_target=5000.0, # 5000性能目标
        constraints={}
    )
    
    print(" START!")
    print(f" 匹配需求:")
    print(f"   功耗预算: {requirement.power_budget}W")
    print(f"   面积预算: {requirement.area_budget}")
    print(f"   性能目标: {requirement.performance_target}")
   

    #随机匹配
    for i in range(n):
        random_result = matcher.random_matching(requirement)
        best_random_score = 0
        if random_result.match_score > best_random_score:
            best_random_score = random_result.match_score
            best_chiplet_num= len(random_result.selected_chiplets)
            best_total_power= random_result.total_power
            best_total_area= random_result.total_area
            best_performance_score= random_result.performance_score
             
        
       
    print(f"\n 随机匹配结果:")
    print(f"   选中芯粒数: {best_chiplet_num}")
    print(f"   总功耗: {best_total_power:.2f}W ")
    print(f"   总面积: {best_total_area:.2f}")
    print(f"   性能分数: {best_performance_score:.2f}")
    print(f"   匹配分数: {best_random_score:.4f}")



    
    # 贪心算法匹配
    for i in range(n):      
        greedy_result = matcher.greedy_matching(requirement)
        best_greddy_score = 0
        if greedy_result.match_score > best_greddy_score:
            best_greddy_score = greedy_result.match_score
            best_chiplet_num= len(greedy_result.selected_chiplets)
            best_total_power= greedy_result.total_power
            best_total_area= greedy_result.total_area
            best_performance_score= greedy_result.performance_score
    
    print(f"\n 贪心算法结果:")
    print(f"   选中芯粒数: {best_chiplet_num}")
    print(f"   总功耗: {best_total_power:.2f}W")
    print(f"   总面积: {best_total_area:.2f}")
    print(f"   性能分数: {best_performance_score:.2f}")
    print(f"   匹配分数: {best_greddy_score:.4f}")

    # 遗传算法匹配

    for i in range(n):
        ga_result = matcher.genetic_algorithm_matching(requirement, population_size=30, generations=50)
        best_ga_score = 0
        if ga_result.match_score > best_ga_score:
            best_ga_score = ga_result.match_score
            best_chiplet_num= len(ga_result.selected_chiplets)
            best_total_power= ga_result.total_power
            best_total_area= ga_result.total_area
            best_performance_score= ga_result.performance_score
                

    print(f"\n 遗传算法结果:")
    print(f"   选中芯粒数: {best_chiplet_num}")
    print(f"   总功耗: {best_total_power:.2f}W")
    print(f"   总面积: {best_total_area:.2f}")
    print(f"   性能分数: {best_performance_score:.2f}")
    print(f"   匹配分数: {best_ga_score:.4f}")

    # 对比结果
    print(f"\n 结果对比:")
    print(f"   随机匹配分数: {best_random_score:.4f}")
    print(f"   贪心算法匹配分数: {best_greddy_score:.4f}")
    print(f"   遗传算法匹配分数: {best_ga_score:.4f}")

    

if __name__ == "__main__":
    main()