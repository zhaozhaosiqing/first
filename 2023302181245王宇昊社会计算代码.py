import numpy as np
import networkx as nx
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ==================== 一、数据预处理与传播权重计算 ====================
class DataPreprocessor:
    def __init__(self, case_data_path, resource_data_path):
        self.case_data = pd.read_csv(case_data_path)
        self.resource_data = pd.read_csv(resource_data_path)
    
    def differential_privacy(self, data, epsilon=1.0):
        """差分隐私脱敏处理"""
        noise = np.random.laplace(0, 1/epsilon, size=data.shape)
        return data + noise
    
    def calculate_transmission_weight(self, contact_duration, protection, susceptibility):
        """计算传播权重"""
        alpha, beta, gamma = 0.4, 0.3, 0.3
        norm_duration = contact_duration / (contact_duration.max() + 1e-8)
        weight = alpha * norm_duration + beta * (1 - protection) + gamma * susceptibility
        return weight
    
    def preprocess(self):
        """数据预处理主流程"""
        # 敏感数据脱敏
        self.case_data['latitude'] = self.differential_privacy(self.case_data['latitude'])
        self.case_data['longitude'] = self.differential_privacy(self.case_data['longitude'])
        
        # 计算传播权重
        self.case_data['transmission_weight'] = self.calculate_transmission_weight(
            self.case_data['contact_duration'],
            self.case_data['protection_level'],
            self.case_data['susceptibility']
        )
        
        # 数据对齐与整合
        merged_data = pd.merge(
            self.case_data,
            self.resource_data[['community_id', 'beds', 'doctors', 'test_points']],
            on='community_id',
            how='left'
        )
        return merged_data

# ==================== 二、改进标签传播算法（传播导向社区发现） ====================
class TransmissionLPA:
    def __init__(self, data, node_col='person_id', weight_col='transmission_weight', community_col='community_id'):
        self.data = data
        self.node_col = node_col
        self.weight_col = weight_col
        self.community_col = community_col
        self.G = self.build_graph()
    
    def build_graph(self):
        """构建传播权重网络（避免重复边）"""
        G = nx.Graph()
        # 添加节点
        nodes = self.data[self.node_col].unique()
        G.add_nodes_from(nodes)
        
        # 构建边：假设数据中每行代表一次接触，两行人互为接触
        edge_records = []
        for _, row in self.data.iterrows():
            node_a = row[self.node_col]
            node_b = row['contact_person_id']
            weight = row[self.weight_col]
            if node_a != node_b and weight > 0.1:
                edge_records.append((node_a, node_b, weight))
        
        # 若存在重复边，取平均权重
        from collections import defaultdict
        edge_dict = defaultdict(list)
        for a, b, w in edge_records:
            if a < b:
                edge_dict[(a, b)].append(w)
            else:
                edge_dict[(b, a)].append(w)
        
        edge_list = [(a, b, np.mean(ws)) for (a, b), ws in edge_dict.items()]
        G.add_weighted_edges_from(edge_list)
        return G
    
    def update_label(self, node):
        """基于传播权重更新节点标签"""
        neighbors = list(self.G.neighbors(node))
        if not neighbors:
            return self.labels[node]
        label_weights = {}
        for neighbor in neighbors:
            label = self.labels[neighbor]
            weight = self.G[node][neighbor]['weight']
            label_weights[label] = label_weights.get(label, 0) + weight
        return max(label_weights, key=label_weights.get)
    
    def fit(self, max_iter=100, epsilon=0.01):
        """训练模型"""
        # 初始化标签
        node_to_community = self.data.set_index(self.node_col)[self.community_col]
        self.labels = {node: node_to_community.get(node, -1) for node in self.G.nodes()}
        
        for it in range(max_iter):
            old_labels = self.labels.copy()
            nodes = list(self.G.nodes())
            np.random.shuffle(nodes)  # 随机顺序更新，提升稳定性
            for node in nodes:
                self.labels[node] = self.update_label(node)
            # 判断收敛
            update_count = sum(1 for k in self.labels if self.labels[k] != old_labels[k])
            if update_count / len(self.labels) < epsilon:
                print(f'LPA 收敛于第 {it+1} 轮迭代')
                break
        
        # 重新映射社区ID为连续整数
        unique_labels = set(self.labels.values())
        community_mapping = {label: i for i, label in enumerate(unique_labels)}
        self.community_assignments = {node: community_mapping[label] for node, label in self.labels.items()}
        return self.community_assignments

# ==================== 三、社区-资源动态匹配模型 ====================
class ResourceAllocator:
    def __init__(self, community_data, resource_data):
        self.community_data = community_data
        self.resource_data = resource_data
    
    def entropy_weight(self, data_matrix):
        """熵权法计算指标权重"""
        # 标准化
        data_norm = (data_matrix - data_matrix.min(axis=0)) / (data_matrix.max(axis=0) - data_matrix.min(axis=0) + 1e-8)
        # 计算比重
        p = data_norm / (data_norm.sum(axis=0) + 1e-8)
        # 计算熵值
        e = -np.sum(p * np.log(p + 1e-8), axis=0) / np.log(len(data_norm))
        # 计算权重
        w = (1 - e) / (1 - e).sum()
        return w
    
    def calculate_demand_score(self):
        """计算社区资源需求得分"""
        indicators = self.community_data[['infection_rate', 'new_case_rate', 'population_density', 'bed_gap_rate']].values
        weights = self.entropy_weight(indicators)
        demand_scores = indicators @ weights
        self.community_data['demand_score'] = demand_scores
        return demand_scores
    
    def optimize_allocation(self, transport_cost_matrix):
        """资源分配优化（贪心算法）"""
        self.calculate_demand_score()
        demand_scores = self.community_data['demand_score'].values
        bed_gap = self.community_data['bed_gap'].values.copy()
        remaining_beds = self.resource_data['total_beds'].values.copy()
        
        n_communities = len(demand_scores)
        n_suppliers = transport_cost_matrix.shape[1]
        allocation = np.zeros((n_communities, n_suppliers))
        
        # 按需求得分降序处理社区
        community_order = np.argsort(-demand_scores)
        for idx in community_order:
            if bed_gap[idx] <= 0:
                continue
            # 选择成本最低且有资源的供给点
            available = np.where(remaining_beds > 0)[0]
            if len(available) == 0:
                break
            costs = transport_cost_matrix[idx, available]
            sorted_suppliers = available[np.argsort(costs)]
            for sup in sorted_suppliers:
                allocate = min(bed_gap[idx], remaining_beds[sup])
                if allocate > 0:
                    allocation[idx, sup] = allocate
                    bed_gap[idx] -= allocate
                    remaining_beds[sup] -= allocate
                if bed_gap[idx] == 0:
                    break
        
        # 计算效益
        total_demand_benefit = np.sum(self.community_data['demand_score'].values * (self.community_data['bed_gap'].values - bed_gap))
        total_transport_cost = np.sum(allocation * transport_cost_matrix)
        total_benefit = total_demand_benefit - 0.1 * total_transport_cost  # lambda=0.1
        
        return allocation, total_benefit

# ==================== 四、SEIR模型模拟与效果评估 ====================
class SEIRModel:
    def __init__(self, beta=0.3, sigma=0.1, gamma=0.05, N=100000):
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.N = N
    
    def seir_equations(self, t, y):
        S, E, I, R = y
        dSdt = -self.beta * S * I / self.N
        dEdt = self.beta * S * I / self.N - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return [dSdt, dEdt, dIdt, dRdt]
    
    def simulate(self, t_span=(0, 100), y0=(0.99, 0.005, 0.005, 0)):
        """模拟疫情传播"""
        solution = solve_ivp(
            fun=self.seir_equations,
            t_span=t_span,
            y0=y0,
            method='RK45',
            t_eval=np.linspace(t_span[0], t_span[1], 100)
        )
        return solution.t, solution.y
    
    def evaluate_intervention(self, beta_intervene=0.2):
        """评估干预效果（降低传染率）"""
        # 保存原始参数
        original_beta = self.beta
        # 无干预模拟
        t1, y1 = self.simulate()
        # 有干预模拟
        self.beta = beta_intervene
        t2, y2 = self.simulate()
        # 恢复参数
        self.beta = original_beta
        
        peak_no = np.max(y1[2])
        peak_in = np.max(y2[2])
        peak_reduction = (peak_no - peak_in) / peak_no
        
        cycle_no = np.sum(y1[2] > 0.001)
        cycle_in = np.sum(y2[2] > 0.001)
        cycle_reduction = (cycle_no - cycle_in) / cycle_no
        
        return {
            'peak_reduction': peak_reduction,
            'cycle_reduction': cycle_reduction,
            't_no_intervene': t1,
            'y_no_intervene': y1,
            't_intervene': t2,
            'y_intervene': y2
        }

# ==================== 五、主流程 ====================
if __name__ == "__main__":
    # 1. 数据预处理（请替换为实际路径或使用模拟数据）
    try:
        preprocessor = DataPreprocessor('case_data.csv', 'resource_data.csv')
        merged_data = preprocessor.preprocess()
        print("数据预处理完成，样本数:", len(merged_data))
    except FileNotFoundError:
        print("警告：未找到数据文件，使用模拟数据示例")
        # 创建模拟数据（仅用于演示）
        merged_data = pd.DataFrame({
            'person_id': range(100),
            'contact_person_id': np.random.randint(0, 100, 100),
            'contact_duration': np.random.uniform(1, 60, 100),
            'protection_level': np.random.uniform(0, 1, 100),
            'susceptibility': np.random.uniform(0, 1, 100),
            'community_id': np.random.randint(0, 5, 100),
            'infection_rate': np.random.uniform(0, 0.3, 100),
            'new_case_rate': np.random.uniform(0, 0.1, 100),
            'population_density': np.random.uniform(100, 10000, 100),
            'beds': np.random.randint(10, 100, 100),
            'bed_gap': np.random.randint(0, 20, 100)
        })
    
    # 2. 社区发现
    lpa = TransmissionLPA(merged_data)
    community_assignments = lpa.fit()
    merged_data['predicted_community'] = merged_data['person_id'].map(community_assignments)
    print("社区发现完成，发现社区数:", len(set(community_assignments.values())))
    
    # 3. 资源分配优化
    community_stats = merged_data.groupby('predicted_community').agg({
        'person_id': 'count',
        'infection_rate': 'mean',
        'new_case_rate': 'mean',
        'population_density': 'mean',
        'beds': 'sum',
        'bed_gap': 'sum'
    }).reset_index()
    community_stats['bed_gap_rate'] = community_stats['bed_gap'] / (community_stats['person_id'] + 1e-8)
    
    resource_suppliers = pd.DataFrame({
        'supplier_id': [0, 1, 2],
        'total_beds': [2000, 3000, 2500]
    })
    
    transport_cost = np.random.uniform(1, 10, (len(community_stats), len(resource_suppliers)))
    allocator = ResourceAllocator(community_stats, resource_suppliers)
    allocation_matrix, total_benefit = allocator.optimize_allocation(transport_cost)
    print("资源分配完成，总效益:", total_benefit)
    
    # 4. SEIR模拟评估
    seir = SEIRModel(beta=0.3, sigma=0.1, gamma=0.05, N=100000)
    intervention_result = seir.evaluate_intervention(beta_intervene=0.2)
    print(f"疫情峰值降低比例：{intervention_result['peak_reduction']:.2%}")
    print(f"传播周期缩短比例：{intervention_result['cycle_reduction']:.2%}")
    
    # 5. 可视化
    plt.figure(figsize=(10, 6))
    t1, y1 = intervention_result['t_no_intervene'], intervention_result['y_no_intervene']
    t2, y2 = intervention_result['t_intervene'], intervention_result['y_intervene']
    plt.plot(t1, y1[2], label='无干预（感染人数）', linewidth=2)
    plt.plot(t2, y2[2], label='有干预（感染人数）', linewidth=2, linestyle='--')
    plt.xlabel('时间（天）')
    plt.ylabel('感染人口比例')
    plt.title('SEIR模型干预效果对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('seir_comparison.png', dpi=300)
    plt.show()