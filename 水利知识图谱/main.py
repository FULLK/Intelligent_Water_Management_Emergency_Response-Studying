# 导入必要的库
import os
from PyPDF2 import PdfReader
import jieba
import re
from transformers import pipeline
from py2neo import Graph, Node, Relationship
import pandas as pd

# 设置Neo4j连接信息
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

# 初始化Neo4j图数据库连接
graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
# 或者在执行查询时指定数据库


# 1. 数据获取与预处理
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def preprocess_text(text):
    # 去除不必要的符号和空格
    cleaned_text = re.sub(r'\s+', ' ', text)
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)

    # 使用jieba进行中文分词
    words = jieba.lcut(cleaned_text)
    return " ".join(words)


def process_pdfs(pdf_dir):
    texts = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            processed_text = preprocess_text(text)  # 预处理
            texts.append(processed_text)
    return texts


# 2. 实体抽取
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-chinese")


def extract_entities(text):
    entities = ner_pipeline(text)
    return entities


def process_entities(texts):
    all_entities = [extract_entities(text) for text in texts]
    return all_entities


# 3. 关系抽取
def extract_relations(entities):
    relations = []
    pattern = r"(\w+)继承于(\w+)"
    matches = re.findall(pattern, " ".join([e['word'] for e in entities]))
    for match in matches:
        relations.append({"source": match[0], "target": match[1], "relation": "继承于"})
    return relations


def process_relations(all_entities):
    all_relations = [extract_relations(entities) for entities in all_entities]
    return all_relations


# 4. 知识融合
entity_mapping = {
    "SWAT": ["SWAT模型", "swat"],
    "VIC": ["VIC模型", "vic"]
    # 添加更多映射...
}


def fuse_entities(entities):
    fused = []
    for entity in entities:
        word = entity['word']
        for key, values in entity_mapping.items():
            if word in values:
                entity['word'] = key
                break
        fused.append(entity)
    return fused


def process_fusion(all_entities):
    fused_entities = [fuse_entities(entities) for entities in all_entities]
    return fused_entities


# 5. 知识存储
def store_knowledge(entities, relations):
    for entity in entities:
        node = Node(entity['entity_type'], name=entity['word'])
        graph.merge(node, entity['entity_type'], "name")

    for relation in relations:
        source_node = graph.nodes.match(relation['source']).first()
        target_node = graph.nodes.match(relation['target']).first()
        if source_node and target_node:
            rel = Relationship(source_node, relation['relation'], target_node)
            graph.create(rel)


def main(pdf_dir):
    # 数据获取与预处理
    texts = process_pdfs(pdf_dir)

    # 实体抽取
    all_entities = process_entities(texts)

    # 关系抽取
    all_relations = process_relations(all_entities)

    # 知识融合
    fused_entities = process_fusion(all_entities)

    # 知识存储
    for i in range(len(fused_entities)):
        store_knowledge(fused_entities[i], all_relations[i])


if __name__ == "__main__":
    # 指定包含PDF文件的目录路径
    pdf_directory = './pdfs'
    main(pdf_directory)