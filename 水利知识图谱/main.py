import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import re
from py2neo import Graph, Node, Relationship
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeGraphBuilder:
    def __init__(self):
        # 初始化BERT模型
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BertModel.from_pretrained('bert-base-chinese')

        # 连接Neo4j数据库
        self.graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

        # 初始化词汇映射表
        self.entity_mapping = self.load_entity_mapping()
        self.relation_mapping = self.load_relation_mapping()

    def load_entity_mapping(self):
        # 加载实体映射表
        # 这里简单示例，实际应该从文件或数据库加载
        return {
            "计算机": ["电脑", "computer"],
            "软件": ["应用程序", "software"]
        }

    def load_relation_mapping(self):
        # 加载关系映射表
        return {
            "包含": ["contains", "包括", "属于"],
            "使用": ["uses", "应用", "采用"]
        }

    def entity_extraction(self, text):
        # 实体抽取
        entities = []

        # 策略1：使用BERT进行命名实体识别
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        bert_embeddings = outputs.last_hidden_state

        # 策略2：基于规则的实体抽取
        patterns = {
            "设备": r"[A-Za-z\u4e00-\u9fa5]+(?:服务器|计算机|设备)",
            "软件": r"[A-Za-z\u4e00-\u9fa5]+(?:软件|系统|平台)"
        }

        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    "text": match.group(),
                    "type": entity_type,
                    "score": 0.8,  # 规则匹配的权重分数
                    "position": match.span()
                })

        # 实体去重和归一化
        entities = self.entity_deduplication(entities)
        return entities

    def entity_deduplication(self, entities):
        # 实体去重
        unique_entities = []
        seen = set()

        for entity in sorted(entities, key=lambda x: x["score"], reverse=True):
            normalized = self.normalize_entity(entity["text"])
            if normalized not in seen:
                seen.add(normalized)
                unique_entities.append(entity)

        return unique_entities

    def normalize_entity(self, entity_text):
        # 实体归一化
        for standard, variants in self.entity_mapping.items():
            if entity_text in variants or entity_text == standard:
                return standard
        return entity_text

    def relation_extraction(self, text, entity1, entity2):
        # 关系抽取
        relations = []

        # 方法1：基于模板的关系抽取
        templates = [
            (r"{}.*?包含.*?{}", "包含"),
            (r"{}.*?使用.*?{}", "使用")
        ]

        for template, relation_type in templates:
            pattern = template.format(re.escape(entity1), re.escape(entity2))
            if re.search(pattern, text):
                relations.append({
                    "type": relation_type,
                    "score": 0.9,  # 模板匹配的权重分数
                    "source": entity1,
                    "target": entity2
                })

                # 方法2：基于距离的关系抽取
            if len(relations) == 0:
                distance = text.find(entity2) - text.find(entity1)
                if 0 < distance < 50:  # 如果两个实体距离较近
                    relations.append({
                        "type": "相关",
                        "score": 0.6,
                        "source": entity1,
                        "target": entity2
                    })

            return relations

        def knowledge_fusion(self, entities, relations):
                # 知识融合
                fused_entities = []
                fused_relations = []

                # 实体融合
                entity_groups = {}
                for entity in entities:
                    normalized = self.normalize_entity(entity["text"])
                    if normalized not in entity_groups:
                        entity_groups[normalized] = []
                    entity_groups[normalized].append(entity)

                # 选择每组得分最高的实体
                for normalized, group in entity_groups.items():
                    fused_entities.append(max(group, key=lambda x: x["score"]))

                # 关系融合
                relation_groups = {}
                for relation in relations:
                    normalized_type = self.normalize_relation(relation["type"])
                    key = (normalized_type, relation["source"], relation["target"])
                    if key not in relation_groups:
                        relation_groups[key] = []
                    relation_groups[key].append(relation)

                # 选择每组得分最高的关系
                for group in relation_groups.values():
                    fused_relations.append(max(group, key=lambda x: x["score"]))

                return fused_entities, fused_relations

        def normalize_relation(self, relation_type):
                # 关系归一化
                for standard, variants in self.relation_mapping.items():
                    if relation_type in variants or relation_type == standard:
                        return standard
                return relation_type

        def store_knowledge(self, entities, relations):
                # 知识存储到Neo4j
                # 清空现有数据
                self.graph.delete_all()

                # 存储实体
                entity_nodes = {}
                for entity in entities:
                    node = Node(entity["type"],
                                name=entity["text"],
                                score=entity["score"])
                    self.graph.create(node)
                    entity_nodes[entity["text"]] = node

                # 存储关系
                for relation in relations:
                    source_node = entity_nodes[relation["source"]]
                    target_node = entity_nodes[relation["target"]]
                    rel = Relationship(source_node, relation["type"], target_node,
                                       score=relation["score"])
                    self.graph.create(rel)

        def build_knowledge_graph(self, text):
                # 主流程：构建知识图谱
                print("开始构建知识图谱...")

                # 1. 实体抽取
                print("正在进行实体抽取...")
                entities = self.entity_extraction(text)

                # 2. 关系抽取
                print("正在进行关系抽取...")
                relations = []
                for i, entity1 in enumerate(entities):
                    for entity2 in entities[i + 1:]:
                        relations.extend(self.relation_extraction(text,
                                                                  entity1["text"],
                                                                  entity2["text"]))

                # 3. 知识融合
                print("正在进行知识融合...")
                fused_entities, fused_relations = self.knowledge_fusion(entities, relations)
                # 4. 知识存储
                print("正在进行知识存储...")
                self.store_knowledge(fused_entities, fused_relations)
                print("知识图谱构建完成！")
                return fused_entities, fused_relations

def main():
    # 使用示例
    kg_builder = KnowledgeGraphBuilder()

    # 示例文本
    text = """
        计算机系统是由硬件和软件组成的。
        操作系统是最重要的系统软件，Windows是最常用的操作系统之一。
        计算机使用CPU进行数据处理，CPU是计算机的核心部件。
        """

    # 构建知识图谱
    entities, relations = kg_builder.build_knowledge_graph(text)

    # 打印结果
    print("\n提取的实体:")
    for entity in entities:
        print(f"实体: {entity['text']}, 类型: {entity['type']}, 得分: {entity['score']}")

    print("\n提取的关系:")
    for relation in relations:
        print(
            f"关系: {relation['source']} --[{relation['type']}]--> {relation['target']}, 得分: {relation['score']}")

if __name__ == "__main__":
    main()