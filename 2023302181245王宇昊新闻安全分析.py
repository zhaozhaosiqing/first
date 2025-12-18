"""
网络安全技术在中美博弈中的关键作用分析
作者：王宇昊
学号：2023302181245
课程：社会计算
"""

import pandas as pd
import numpy as np
import jieba
import jieba.analyse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec, KeyedVectors
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Set
import warnings
import requests
import json
from datetime import datetime, timedelta
from collections import Counter
import re

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 一、数据获取与预处理模块 ====================
class DataCollector:
    """数据收集器 - 模拟多源数据获取"""
    
    def __init__(self, use_simulated=True):
        self.use_simulated = use_simulated
        self.game_keywords = {
            'zh': ['安全博弈', '技术竞争', '网络安全', 'AI安全', '量子加密', 
                   '数据安全', '国防安全', '技术制裁', '专利竞争', '关键基础设施',
                   '网络空间', '大国竞争', '技术自主', '供应链安全', '信息战'],
            'en': ['cybersecurity', 'technology competition', 'AI safety', 'quantum encryption',
                   'data security', 'national defense', 'tech sanctions', 'patent war',
                   'critical infrastructure', 'cyberspace', 'great power competition']
        }
        
        # 网络安全领域停用词
        self.security_stopwords = {
            'zh': set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '说', '要', '也']),
            'en': set(stopwords.words('english')).union({'said', 'like', 'would', 'could', 'one', 'two', 'three'})
        }
    
    def simulate_data(self, n_samples=2000) -> pd.DataFrame:
        """生成模拟实验数据"""
        np.random.seed(42)
        
        # 中文文本模板
        zh_templates = [
            "中美在{tech}领域的技术博弈日趋激烈，双方都在加大研发投入",
            "中国在{tech}方面取得重要突破，打破了{tech}的技术垄断",
            "美国对华{tech}实施制裁，限制相关技术出口",
            "{tech}成为中美网络安全博弈的新焦点",
            "专家认为，{tech}将重塑未来网络空间安全格局",
            "中国{tech}专利申请量大幅增长，缩小与美国的差距",
            "{tech}技术在国防领域的应用引发广泛关注",
            "中美{tech}标准制定权的竞争进入关键阶段"
        ]
        
        # 英文文本模板
        en_templates = [
            "US-China competition in {tech} intensifies as both sides increase R&D investment",
            "China makes breakthrough in {tech}, challenging US dominance",
            "US imposes sanctions on Chinese {tech} companies over security concerns",
            "{tech} emerges as new battleground in US-China cybersecurity competition",
            "Experts believe {tech} will reshape future cybersecurity landscape",
            "China's {tech} patent applications surge, narrowing gap with US",
            "Military applications of {tech} raise concerns in Washington",
            "US and China compete for {tech} standard-setting authority"
        ]
        
        # 技术领域
        tech_areas = ['AI安全', '量子加密', '数据跨境安全', '关键基础设施防护', 
                     '零信任架构', '区块链安全', '5G安全', '物联网安全']
        
        # 领域关键词
        domain_keywords = {
            '行业发展': ['市场', '产业', '企业', '经济', '商业', '投资'],
            '人才培养': ['人才', '教育', '培训', '专家', '团队', '高校'],
            '科研突破': ['研发', '专利', '创新', '技术', '突破', '实验室'],
            '国防军事': ['国防', '军事', '军队', '作战', '防御', '安全']
        }
        
        data_records = []
        sentiment_labels = {'positive': 1, 'negative': -1, 'neutral': 0}
        
        for i in range(n_samples):
            lang = np.random.choice(['zh', 'en'], p=[0.6, 0.4])
            tech = np.random.choice(tech_areas)
            
            if lang == 'zh':
                template = np.random.choice(zh_templates)
                content = template.format(tech=tech)
                # 添加领域关键词
                if np.random.random() > 0.3:
                    domain = np.random.choice(list(domain_keywords.keys()))
                    keyword = np.random.choice(domain_keywords[domain])
                    content += f"，这对{domain}产生重要影响"
            else:
                template = np.random.choice(en_templates)
                tech_en = tech.replace('AI安全', 'AI safety').replace('量子加密', 'quantum encryption')
                content = template.format(tech=tech_en)
            
            # 生成情感标签（基于内容和关键词）
            sentiment = 0
            if any(word in content for word in ['突破', '增长', '领先', '优势', 'success', 'breakthrough']):
                sentiment = 1
            elif any(word in content for word in ['制裁', '限制', '威胁', '挑战', 'sanction', 'threat']):
                sentiment = -1
            
            data_records.append({
                'id': i,
                'language': lang,
                'content': content,
                'tech_area': tech,
                'source': np.random.choice(['媒体', '官方文件', '学术论文', '行业报告']),
                'date': (datetime(2022, 1, 1) + timedelta(days=np.random.randint(0, 1095))).strftime('%Y-%m-%d'),
                'sentiment_label': sentiment
            })
        
        return pd.DataFrame(data_records)
    
    def collect_real_data(self) -> pd.DataFrame:
        """收集真实数据（API调用示例）"""
        # 这里可以添加真实数据收集逻辑
        # 例如：调用新闻API、爬取官方网站等
        print("使用模拟数据，如需真实数据请配置API接口")
        return self.simulate_data()


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        # 初始化jieba
        jieba.initialize()
        # 加载自定义词典
        self.load_custom_dict()
        
        # 博弈核心主题词
        self.game_core_words = {
            'zh': ['安全博弈', '技术竞争', '网络安全', '大国竞争', '网络空间', '技术自主',
                  '供应链安全', '信息战', '科技战', '标准制定', '技术封锁'],
            'en': ['cybersecurity competition', 'technology rivalry', 'great power competition',
                  'cyberspace', 'tech autonomy', 'supply chain security', 'information warfare',
                  'tech war', 'standard setting', 'technology blockade']
        }
    
    def load_custom_dict(self):
        """加载网络安全领域自定义词典"""
        custom_dict = {
            'AI安全': 100, '量子加密': 100, '数据跨境': 100, '零信任': 100,
            '区块链安全': 100, '5G安全': 100, '物联网安全': 100,
            '网络空间安全': 100, '关键基础设施': 100, '供应链安全': 100
        }
        for word, freq in custom_dict.items():
            jieba.add_word(word, freq=freq)
    
    def preprocess_chinese(self, text: str) -> List[str]:
        """中文文本预处理"""
        # 清洗文本
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 使用jieba进行分词和关键词提取
        words = jieba.lcut(text)
        
        # 过滤：长度>1，非停用词，包含中文字符或英文单词
        filtered_words = []
        for word in words:
            if (len(word) > 1 and 
                not word.isspace() and
                (re.search(r'[\u4e00-\u9fa5]', word) or word.isalpha())):
                filtered_words.append(word)
        
        return filtered_words
    
    def preprocess_english(self, text: str) -> List[str]:
        """英文文本预处理"""
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        
        # 分词
        words = word_tokenize(text)
        
        # 过滤停用词和短词
        english_stopwords = set(stopwords.words('english'))
        filtered_words = [word for word in words 
                         if (len(word) > 2 and 
                             word not in english_stopwords and
                             word.isalpha())]
        
        return filtered_words
    
    def calculate_game_relevance(self, words: List[str], language: str) -> float:
        """计算文本博弈关联度"""
        if not words:
            return 0.0
        
        core_words = self.game_core_words[language]
        matches = 0
        for word in words:
            # 检查是否为核心词的组成部分
            for core in core_words:
                if core in word or word in core:
                    matches += 1
                    break
        
        relevance = matches / len(words)
        # 应用非线性映射（突出高关联度文本）
        return min(1.0, relevance * 2)
    
    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """批量处理数据"""
        processed_records = []
        
        for _, row in df.iterrows():
            lang = row['language']
            content = row['content']
            
            # 分词
            if lang == 'zh':
                words = self.preprocess_chinese(content)
            else:
                words = self.preprocess_english(content)
            
            if not words:  # 跳过空文本
                continue
            
            # 计算博弈关联度
            relevance = self.calculate_game_relevance(words, lang)
            
            # 提取文本特征
            text_length = len(words)
            unique_words = len(set(words))
            lexical_diversity = unique_words / text_length if text_length > 0 else 0
            
            processed_records.append({
                'id': row['id'],
                'language': lang,
                'original_content': content,
                'words': words,
                'word_count': text_length,
                'unique_words': unique_words,
                'lexical_diversity': lexical_diversity,
                'game_relevance': relevance,
                'tech_area': row.get('tech_area', ''),
                'source': row.get('source', ''),
                'date': row.get('date', ''),
                'sentiment_label': row.get('sentiment_label', 0)
            })
        
        result_df = pd.DataFrame(processed_records)
        
        # 过滤低关联度文本
        result_df = result_df[result_df['game_relevance'] >= 0.1].reset_index(drop=True)
        
        print(f"数据预处理完成：{len(result_df)} 条有效文本")
        print(f"平均博弈关联度：{result_df['game_relevance'].mean():.3f}")
        print(f"平均文本长度：{result_df['word_count'].mean():.1f} 词")
        
        return result_df


# ==================== 二、博弈焦点识别算法 ====================
class GameFocusAnalyzer:
    """博弈焦点分析器"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.languages = df['language'].unique()
        self.results = {}
        
    def improved_tfidf(self, top_k=20, min_df=5) -> pd.DataFrame:
        """改进的TF-IDF算法（融合博弈关联度）"""
        all_keywords = []
        
        for lang in self.languages:
            lang_df = self.df[self.df['language'] == lang]
            
            # 准备文本（词列表转空格分隔字符串）
            texts = [' '.join(words) for words in lang_df['words']]
            relevances = lang_df['game_relevance'].values
            
            if not texts:
                continue
            
            # 计算传统TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                min_df=min_df,
                max_df=0.8,
                ngram_range=(1, 2)  # 考虑1-2元语法
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # 应用博弈关联度修正
                # 每个词的权重 = sum(TF-IDF * 博弈关联度) / sum(博弈关联度)
                weighted_scores = np.zeros(len(feature_names))
                
                for i, text_vec in enumerate(tfidf_matrix):
                    # 获取该文本中每个词的TF-IDF值
                    row = text_vec.toarray().flatten()
                    # 乘以该文本的博弈关联度
                    weighted_row = row * relevances[i]
                    weighted_scores += weighted_row
                
                # 归一化
                if weighted_scores.sum() > 0:
                    weighted_scores = weighted_scores / weighted_scores.sum()
                
                # 获取top-k关键词
                top_indices = np.argsort(weighted_scores)[-top_k:][::-1]
                lang_keywords = []
                
                for idx in top_indices:
                    if weighted_scores[idx] > 0:
                        lang_keywords.append({
                            'keyword': feature_names[idx],
                            'score': weighted_scores[idx],
                            'language': lang
                        })
                
                all_keywords.extend(lang_keywords)
                
            except ValueError as e:
                print(f"语言 {lang} 的TF-IDF计算失败：{e}")
                continue
        
        result_df = pd.DataFrame(all_keywords)
        if not result_df.empty:
            result_df = result_df.sort_values('score', ascending=False).head(top_k * 2)
            self.results['keywords'] = result_df
        
        return result_df
    
    def train_word2vec(self, vector_size=100, window=5, min_count=2):
        """训练Word2Vec模型"""
        all_sentences = self.df['words'].tolist()
        
        if len(all_sentences) < 100:
            print("训练数据不足，跳过Word2Vec训练")
            return None
        
        print(f"训练Word2Vec模型，语料大小：{len(all_sentences)}")
        model = Word2Vec(
            sentences=all_sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            epochs=10
        )
        
        self.w2v_model = model
        return model
    
    def get_text_vector(self, words: List[str]) -> np.ndarray:
        """获取文本向量（Word2Vec均值）"""
        if not hasattr(self, 'w2v_model') or self.w2v_model is None:
            return np.zeros(100)
        
        vectors = []
        for word in words:
            if word in self.w2v_model.wv:
                vectors.append(self.w2v_model.wv[word])
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(100)
    
    def sentiment_analysis(self) -> pd.DataFrame:
        """情感分析（无监督+弱监督方法）"""
        
        # 1. 构建领域情感词典
        security_lexicon = self._build_security_lexicon()
        
        # 2. 计算词典匹配得分
        def lexicon_score(words):
            score = 0
            for word in words:
                if word in security_lexicon['positive']:
                    score += 1
                elif word in security_lexicon['negative']:
                    score -= 1
            return score / len(words) if len(words) > 0 else 0
        
        self.df['lexicon_score'] = self.df['words'].apply(lexicon_score)
        
        # 3. 使用TextBlob进行英文情感分析（如可用）
        try:
            from textblob import TextBlob
            def blob_sentiment(text):
                blob = TextBlob(text)
                return blob.sentiment.polarity
            
            # 仅对英文文本应用
            self.df['blob_score'] = self.df.apply(
                lambda x: blob_sentiment(x['original_content']) if x['language'] == 'en' else 0,
                axis=1
            )
        except ImportError:
            self.df['blob_score'] = 0
            print("TextBlob未安装，跳过英文情感分析")
        
        # 4. 结合多种分数（如果已有标注，可训练模型）
        if 'sentiment_label' in self.df.columns and self.df['sentiment_label'].notna().any():
            # 使用逻辑回归结合特征
            X = []
            for _, row in self.df.iterrows():
                # 特征：词典分数 + 文本向量
                vec = self.get_text_vector(row['words'])
                features = np.concatenate([[row['lexicon_score']], vec])
                X.append(features)
            
            X = np.array(X)
            y = self.df['sentiment_label'].values
            
            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 训练模型
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            model = LogisticRegression(max_iter=1000, class_weight='balanced')
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            print("\n情感分析模型性能：")
            print(classification_report(y_test, y_pred))
            
            # 对整个数据集预测
            self.df['predicted_sentiment'] = model.predict(X_scaled)
        else:
            # 无监督方法：基于词典分数和TextBlob分数
            self.df['combined_score'] = (
                self.df['lexicon_score'] * 0.7 + 
                self.df['blob_score'] * 0.3
            )
            
            # 转换为离散标签
            def score_to_label(score):
                if score > 0.1:
                    return 1  # 正向
                elif score < -0.1:
                    return -1  # 负向
                else:
                    return 0  # 中性
            
            self.df['predicted_sentiment'] = self.df['combined_score'].apply(score_to_label)
        
        return self.df[['original_content', 'predicted_sentiment', 'lexicon_score']]
    
    def _build_security_lexicon(self) -> Dict[str, Set[str]]:
        """构建网络安全领域情感词典"""
        lexicon = {
            'positive': {
                '突破', '创新', '领先', '安全', '防护', '优势', '进步', '发展',
                '成功', '强大', '可靠', '稳定', '先进', '自主', '可控'
            },
            'negative': {
                '威胁', '制裁', '攻击', '漏洞', '风险', '劣势', '挑战', '危机',
                '失败', '弱点', '入侵', '破坏', '泄露', '危险', '威胁'
            }
        }
        
        # 添加英文词
        lexicon['positive'].update({
            'breakthrough', 'innovation', 'leading', 'secure', 'protection',
            'advantage', 'progress', 'development', 'success', 'strong'
        })
        
        lexicon['negative'].update({
            'threat', 'sanction', 'attack', 'vulnerability', 'risk',
            'disadvantage', 'challenge', 'crisis', 'failure', 'weakness'
        })
        
        return lexicon
    
    def analyze_temporal_trends(self) -> pd.DataFrame:
        """分析时间趋势"""
        if 'date' not in self.df.columns:
            print("无时间数据，跳过趋势分析")
            return pd.DataFrame()
        
        # 确保日期格式
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        self.df = self.df.dropna(subset=['date'])
        
        # 按季度聚合
        self.df['quarter'] = self.df['date'].dt.to_period('Q')
        
        # 分析技术领域热度趋势
        if 'tech_area' in self.df.columns:
            trend_data = []
            for quarter in sorted(self.df['quarter'].unique()):
                quarter_df = self.df[self.df['quarter'] == quarter]
                tech_counts = quarter_df['tech_area'].value_counts()
                
                for tech, count in tech_counts.items():
                    trend_data.append({
                        'quarter': str(quarter),
                        'tech_area': tech,
                        'count': count,
                        'proportion': count / len(quarter_df)
                    })
            
            return pd.DataFrame(trend_data)
        
        return pd.DataFrame()


# ==================== 三、影响传导关联挖掘 ====================
class ImpactTransmissionAnalyzer:
    """影响传导分析器"""
    
    def __init__(self, df: pd.DataFrame, tech_keywords: List[str]):
        self.df = df
        self.tech_keywords = tech_keywords
        
        # 定义领域关键词
        self.domain_keywords = {
            '行业发展': ['市场', '产业', '企业', '经济', '商业', '投资', '市值', '营收', '利润',
                      'market', 'industry', 'enterprise', 'economy', 'business'],
            '人才培养': ['人才', '教育', '培训', '专家', '团队', '高校', '课程', '技能', '招聘',
                       'talent', 'education', 'training', 'expert', 'university'],
            '科研突破': ['研发', '专利', '创新', '技术', '突破', '实验室', '论文', '成果', '发明',
                       'R&D', 'patent', 'innovation', 'technology', 'breakthrough'],
            '国防军事': ['国防', '军事', '军队', '作战', '防御', '安全', '战略', '武器', '装备',
                       'defense', 'military', 'army', 'operation', 'security']
        }
    
    def build_transaction_data(self) -> pd.DataFrame:
        """构建事务数据"""
        transactions = []
        
        for _, row in self.df.iterrows():
            words = set(row['words'])
            items = []
            
            # 技术关键词
            for tech in self.tech_keywords:
                # 检查技术关键词是否在文本中（支持部分匹配）
                tech_words = tech.split()
                if any(tech_word in words for tech_word in tech_words if len(tech_word) > 1):
                    items.append(f'TECH_{tech}')
            
            # 领域关键词
            for domain, keywords in self.domain_keywords.items():
                if any(keyword in words for keyword in keywords):
                    items.append(f'DOMAIN_{domain}')
            
            if items:  # 仅保留非空事务
                transactions.append(items)
        
        # 转换为one-hot编码
        all_items = sorted(set([item for trans in transactions for item in trans]))
        trans_matrix = pd.DataFrame(0, index=range(len(transactions)), columns=all_items)
        
        for i, items in enumerate(transactions):
            trans_matrix.loc[i, items] = 1
        
        print(f"构建事务数据：{len(transactions)} 条事务，{len(all_items)} 个唯一项")
        return trans_matrix
    
    def mine_association_rules(self, min_support=0.05, min_confidence=0.6) -> pd.DataFrame:
        """挖掘关联规则"""
        trans_matrix = self.build_transaction_data()
        
        if trans_matrix.empty or len(trans_matrix.columns) < 2:
            print("事务数据不足，无法挖掘关联规则")
            return pd.DataFrame()
        
        # 挖掘频繁项集
        frequent_itemsets = apriori(
            trans_matrix,
            min_support=min_support,
            use_colnames=True,
            max_len=3  # 限制项集大小
        )
        
        if frequent_itemsets.empty:
            print("未找到频繁项集")
            return pd.DataFrame()
        
        # 生成关联规则
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )
        
        # 筛选技术->领域的规则
        def is_tech_to_domain_rule(antecedents, consequents):
            antecedents_str = str(antecedents)
            consequents_str = str(consequents)
            return ('TECH_' in antecedents_str and 'DOMAIN_' in consequents_str)
        
        tech_domain_rules = rules[
            rules.apply(lambda x: is_tech_to_domain_rule(x['antecedents'], x['consequents']), axis=1)
        ]
        
        # 格式化规则
        formatted_rules = []
        for _, rule in tech_domain_rules.iterrows():
            # 提取技术名称
            tech_items = [item for item in rule['antecedents'] if 'TECH_' in item]
            tech_names = [item.replace('TECH_', '') for item in tech_items]
            
            # 提取领域名称
            domain_items = [item for item in rule['consequents'] if 'DOMAIN_' in item]
            domain_names = [item.replace('DOMAIN_', '') for item in domain_items]
            
            if tech_names and domain_names:
                formatted_rules.append({
                    'technology': ' & '.join(tech_names),
                    'domain': ' & '.join(domain_names),
                    'support': rule['support'],
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'coverage': len(tech_domain_rules) / len(rules) if len(rules) > 0 else 0
                })
        
        result_df = pd.DataFrame(formatted_rules)
        if not result_df.empty:
            result_df = result_df.sort_values(['confidence', 'support'], ascending=False)
        
        return result_df
    
    def calculate_impact_strength(self) -> Dict[str, float]:
        """计算影响强度矩阵"""
        impact_matrix = {}
        
        trans_matrix = self.build_transaction_data()
        tech_columns = [col for col in trans_matrix.columns if 'TECH_' in col]
        domain_columns = [col for col in trans_matrix.columns if 'DOMAIN_' in col]
        
        for tech in tech_columns:
            tech_name = tech.replace('TECH_', '')
            impact_matrix[tech_name] = {}
            
            for domain in domain_columns:
                domain_name = domain.replace('DOMAIN_', '')
                
                # 计算共现概率 P(domain|tech)
                tech_domain_cooccur = ((trans_matrix[tech] == 1) & (trans_matrix[domain] == 1)).sum()
                tech_occur = (trans_matrix[tech] == 1).sum()
                
                if tech_occur > 0:
                    impact_strength = tech_domain_cooccur / tech_occur
                    impact_matrix[tech_name][domain_name] = impact_strength
        
        return impact_matrix


# ==================== 四、可视化与报告生成 ====================
class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self):
        self.colors = {
            'tech': '#FF6B6B',  # 红色系 - 技术
            'domain': '#4ECDC4',  # 青色系 - 领域
            'positive': '#2E8B57',  # 绿色 - 正向
            'negative': '#DC143C',  # 红色 - 负向
            'neutral': '#4682B4'  # 蓝色 - 中性
        }
    
    def plot_keyword_heatmap(self, keywords_df: pd.DataFrame, top_n=20):
        """绘制关键词热度图"""
        if keywords_df.empty:
            print("无关键词数据")
            return
        
        # 按语言分组
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        for idx, lang in enumerate(['zh', 'en']):
            lang_df = keywords_df[keywords_df['language'] == lang].head(top_n//2)
            
            if lang_df.empty:
                continue
            
            ax = axes[idx]
            y_pos = np.arange(len(lang_df))
            
            ax.barh(y_pos, lang_df['score'], color=self.colors['tech'], alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(lang_df['keyword'], fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel('修正TF-IDF得分', fontsize=12)
            ax.set_title(f'{lang.upper()}文本核心博弈焦点', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        
        plt.suptitle('中美网络安全博弈核心技术焦点', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('keyword_focus.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sentiment_distribution(self, sentiment_df: pd.DataFrame):
        """绘制情感分布图"""
        if sentiment_df.empty or 'predicted_sentiment' not in sentiment_df.columns:
            print("无情感到数据")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 总体情感分布
        sentiment_counts = sentiment_df['predicted_sentiment'].value_counts().sort_index()
        sentiment_labels = {1: '正向', -1: '负向', 0: '中性'}
        
        colors = [self.colors['positive'], self.colors['negative'], self.colors['neutral']]
        axes[0].pie(sentiment_counts.values, labels=[sentiment_labels.get(i, i) for i in sentiment_counts.index],
                   colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('总体情感分布', fontsize=14, fontweight='bold')
        
        # 按语言的情感分布
        if 'language' in sentiment_df.columns:
            sentiment_by_lang = pd.crosstab(sentiment_df['language'], 
                                           sentiment_df['predicted_sentiment'])
            sentiment_by_lang.plot(kind='bar', ax=axes[1], 
                                  color=[self.colors['positive'], self.colors['negative'], self.colors['neutral']])
            axes[1].set_title('分语言情感分布', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('语言')
            axes[1].set_ylabel('文本数量')
            axes[1].legend(['正向', '负向', '中性'])
            axes[1].tick_params(axis='x', rotation=0)
        
        plt.suptitle('网络安全博弈文本情感分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_impact_network(self, rules_df: pd.DataFrame, min_confidence=0.5):
        """绘制影响传导网络图"""
        if rules_df.empty:
            print("无关联规则数据")
            return
        
        # 筛选高置信度规则
        filtered_rules = rules_df[rules_df['confidence'] >= min_confidence].head(20)
        
        if filtered_rules.empty:
            print("无符合条件的关联规则")
            return
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点和边
        for _, rule in filtered_rules.iterrows():
            tech = rule['technology']
            domain = rule['domain']
            confidence = rule['confidence']
            
            G.add_node(tech, node_type='tech', size=confidence*500)
            G.add_node(domain, node_type='domain', size=confidence*300)
            G.add_edge(tech, domain, weight=confidence, confidence=confidence)
        
        if len(G.nodes()) == 0:
            print("网络图节点为空")
            return
        
        # 布局
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # 绘制
        plt.figure(figsize=(12, 10))
        
        # 按节点类型设置颜色和大小
        tech_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('node_type') == 'tech']
        domain_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('node_type') == 'domain']
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, nodelist=tech_nodes,
                              node_color=self.colors['tech'], 
                              node_size=[G.nodes[node]['size'] for node in tech_nodes],
                              alpha=0.9, label='核心技术')
        
        nx.draw_networkx_nodes(G, pos, nodelist=domain_nodes,
                              node_color=self.colors['domain'], 
                              node_size=[G.nodes[node]['size'] for node in domain_nodes],
                              alpha=0.9, label='影响领域')
        
        # 绘制边（宽度表示置信度）
        edges = G.edges(data=True)
        edge_weights = [d['weight']*3 for (u, v, d) in edges]
        
        nx.draw_networkx_edges(G, pos, edgelist=edges,
                              width=edge_weights,
                              alpha=0.6,
                              edge_color='gray',
                              arrowsize=15,
                              arrowstyle='->')
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        # 添加图例
        plt.legend(scatterpoints=1, frameon=True, fontsize=12)
        
        plt.title('网络安全技术-领域影响传导网络\n（连线粗细表示关联置信度）', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('impact_transmission_network.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_temporal_trends(self, trends_df: pd.DataFrame):
        """绘制时间趋势图"""
        if trends_df.empty:
            print("无趋势数据")
            return
        
        plt.figure(figsize=(14, 8))
        
        # 透视表：季度×技术领域
        pivot_df = trends_df.pivot_table(index='quarter', columns='tech_area', 
                                        values='proportion', aggfunc='sum')
        
        # 绘制堆叠面积图
        pivot_df.plot(kind='area', stacked=True, alpha=0.7, 
                     colormap='tab20', figsize=(14, 8))
        
        plt.title('中美网络安全博弈技术焦点时间趋势', fontsize=16, fontweight='bold')
        plt.xlabel('季度', fontsize=12)
        plt.ylabel('热度比例', fontsize=12)
        plt.legend(title='技术领域', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('temporal_trends.png', dpi=300, bbox_inches='tight')
        plt.show()


# ==================== 五、主实验流程 ====================
def main():
    """主实验流程"""
    print("=" * 70)
    print("网络安全技术在中美博弈中的关键作用分析")
    print("基于多源数据的焦点识别与影响传导研究")
    print("=" * 70)
    
    # 1. 数据收集与预处理
    print("\n阶段1：数据收集与预处理")
    print("-" * 40)
    
    collector = DataCollector(use_simulated=True)
    if collector.use_simulated:
        print("使用模拟数据...")
        raw_data = collector.simulate_data(n_samples=2000)
    else:
        print("收集真实数据...")
        raw_data = collector.collect_real_data()
    
    print(f"原始数据规模：{len(raw_data)} 条")
    
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process_batch(raw_data)
    
    # 2. 博弈焦点识别
    print("\n阶段2：博弈焦点识别与分析")
    print("-" * 40)
    
    analyzer = GameFocusAnalyzer(processed_data)
    
    # 2.1 改进TF-IDF关键词提取
    print("执行改进TF-IDF算法...")
    keywords_df = analyzer.improved_tfidf(top_k=25)
    
    if not keywords_df.empty:
        print("\n核心博弈技术焦点TOP10：")
        for i, (_, row) in enumerate(keywords_df.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['keyword']:15s} (得分：{row['score']:.4f}, 语言：{row['language']})")
    
    # 2.2 情感分析
    print("\n执行情感分析...")
    sentiment_df = analyzer.sentiment_analysis()
    
    if not sentiment_df.empty:
        sentiment_stats = sentiment_df['predicted_sentiment'].value_counts()
        print(f"情感分布：正向 {sentiment_stats.get(1, 0)} 条，"
              f"负向 {sentiment_stats.get(-1, 0)} 条，"
              f"中性 {sentiment_stats.get(0, 0)} 条")
    
    # 2.3 时间趋势分析
    print("\n分析时间趋势...")
    trends_df = analyzer.analyze_temporal_trends()
    
    # 3. 影响传导分析
    print("\n阶段3：影响传导关联挖掘")
    print("-" * 40)
    
    if not keywords_df.empty:
        tech_keywords = keywords_df['keyword'].head(15).tolist()
        impact_analyzer = ImpactTransmissionAnalyzer(processed_data, tech_keywords)
        
        # 3.1 挖掘关联规则
        print("挖掘技术-领域关联规则...")
        rules_df = impact_analyzer.mine_association_rules(min_support=0.03, min_confidence=0.5)
        
        if not rules_df.empty:
            print("\nTop 5 影响传导关联规则：")
            for i, (_, rule) in enumerate(rules_df.head(5).iterrows(), 1):
                print(f"{i}. {rule['technology']} → {rule['domain']} "
                      f"(置信度：{rule['confidence']:.3f}, 支持度：{rule['support']:.3f})")
        
        # 3.2 计算影响强度矩阵
        print("\n计算影响强度矩阵...")
        impact_matrix = impact_analyzer.calculate_impact_strength()
        
        # 打印影响强度TOP3
        print("\n技术对领域影响强度TOP3：")
        impact_records = []
        for tech, domains in impact_matrix.items():
            for domain, strength in domains.items():
                impact_records.append((tech, domain, strength))
        
        impact_records.sort(key=lambda x: x[2], reverse=True)
        for i, (tech, domain, strength) in enumerate(impact_records[:3], 1):
            print(f"{i}. {tech} → {domain}: {strength:.3f}")
    
    # 4. 可视化结果
    print("\n阶段4：结果可视化")
    print("-" * 40)
    
    visualizer = ResultVisualizer()
    
    # 4.1 关键词热度图
    if not keywords_df.empty:
        visualizer.plot_keyword_heatmap(keywords_df, top_n=20)
        print("✓ 关键词热度图已保存")
    
    # 4.2 情感分布图
    if not sentiment_df.empty:
        visualizer.plot_sentiment_distribution(sentiment_df)
        print("✓ 情感分布图已保存")
    
    # 4.3 影响传导网络图
    if 'rules_df' in locals() and not rules_df.empty:
        visualizer.plot_impact_network(rules_df, min_confidence=0.4)
        print("✓ 影响传导网络图已保存")
    
    # 4.4 时间趋势图
    if 'trends_df' in locals() and not trends_df.empty:
        visualizer.plot_temporal_trends(trends_df)
        print("✓ 时间趋势图已保存")
    
    # 5. 结果保存与报告
    print("\n阶段5：结果保存")
    print("-" * 40)
    
    # 保存关键结果到CSV
    if 'keywords_df' in locals() and not keywords_df.empty:
        keywords_df.to_csv('game_focus_keywords.csv', index=False, encoding='utf-8-sig')
        print("✓ 博弈焦点关键词已保存至 game_focus_keywords.csv")
    
    if 'sentiment_df' in locals() and not sentiment_df.empty:
        sentiment_summary = sentiment_df.copy()
        sentiment_summary.to_csv('sentiment_analysis.csv', index=False, encoding='utf-8-sig')
        print("✓ 情感分析结果已保存至 sentiment_analysis.csv")
    
    if 'rules_df' in locals() and not rules_df.empty:
        rules_df.to_csv('impact_transmission_rules.csv', index=False, encoding='utf-8-sig')
        print("✓ 影响传导关联规则已保存至 impact_transmission_rules.csv")
    
    if 'trends_df' in locals() and not trends_df.empty:
        trends_df.to_csv('temporal_trends.csv', index=False, encoding='utf-8-sig')
        print("✓ 时间趋势数据已保存至 temporal_trends.csv")
    
    # 6. 实验总结
    print("\n" + "=" * 70)
    print("实验总结报告")
    print("=" * 70)
    
    print(f"\n1. 数据处理结果：")
    print(f"   - 有效文本数量：{len(processed_data)} 条")
    print(f"   - 平均博弈关联度：{processed_data['game_relevance'].mean():.3f}")
    print(f"   - 平均文本长度：{processed_data['word_count'].mean():.1f} 词")
    
    print(f"\n2. 博弈焦点识别结果：")
    if 'keywords_df' in locals() and not keywords_df.empty:
        top_tech = keywords_df.head(3)['keyword'].tolist()
        print(f"   - 核心博弈焦点：{', '.join(top_tech)}")
        print(f"   - 识别关键词数量：{len(keywords_df)} 个")
    
    print(f"\n3. 情感分析结果：")
    if 'sentiment_df' in locals() and not sentiment_df.empty:
        sentiment_counts = sentiment_df['predicted_sentiment'].value_counts()
        pos_pct = sentiment_counts.get(1, 0) / len(sentiment_df) * 100
        neg_pct = sentiment_counts.get(-1, 0) / len(sentiment_df) * 100
        print(f"   - 正向情感占比：{pos_pct:.1f}%")
        print(f"   - 负向情感占比：{neg_pct:.1f}%")
        print(f"   - 中性情感占比：{100 - pos_pct - neg_pct:.1f}%")
    
    print(f"\n4. 影响传导分析结果：")
    if 'rules_df' in locals() and not rules_df.empty:
        print(f"   - 发现关联规则：{len(rules_df)} 条")
        if len(rules_df) > 0:
            avg_confidence = rules_df['confidence'].mean()
            max_confidence = rules_df['confidence'].max()
            print(f"   - 平均置信度：{avg_confidence:.3f}")
            print(f"   - 最高置信度：{max_confidence:.3f}")
    
    print(f"\n5. 可视化输出：")
    print(f"   - 已生成4张分析图表")
    print(f"   - 已保存4个数据文件")
    
    print("\n" + "=" * 70)
    print("实验完成！所有结果文件已保存至当前目录")
    print("=" * 70)


if __name__ == "__main__":
    # 运行完整实验流程
    main()