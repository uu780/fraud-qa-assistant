# 🛡️ 防诈骗智能问答助手 (Fraud QA Assistant)
Python 3.10+ FastAPI License: MIT

基于 RAG (Retrieval-Augmented Generation) 技术的防诈骗智能问答系统。通过构建专业的防诈知识库，结合多路检索、重排序 (Reranking) 和语义缓存，利用千问大模型为用户提供精准、专业的防诈骗指导。

## ✨ 功能特性

### 检索增强
- **混合检索**：结合向量检索 (Dense) 与 BM25 关键词检索 (Sparse)，兼顾语义理解与精准匹配
- **多查询扩展**：利用 LLM 将用户问题重述为多个视角，提升检索覆盖率
- **RRF 融合**：采用倒数排名融合算法，整合多路检索结果
- **BGE Reranker**：引入 Cross-Encoder 重排序模型，对初筛文档进行二次精排

### 高性能响应
- **多级缓存**：集成 FAISS 语义缓存与精确匹配缓存，相同或相似问题秒级响应
- **流式输出**：基于 FastAPI 的 StreamingResponse，实现答案实时打字机效果

### 量化评测
- 集成 Ragas 评测框架，从忠实度、相关性等维度进行量化分析

## 📁 项目结构
fraud-qa-assistant/
├── app/ # 核心应用代码
│ ├── cache/ # 缓存模块
│ │ ├── cache_manager.py
│ │ ├── response_cache.py
│ │ └── semantic_cache.py
│ ├── core.py # RAG 核心流水线
│ ├── evaluation.py # Ragas 评测模块
│ └── index.py # 向量索引构建
├── data/ # 数据文件
│ └── raw/ # 原始知识库 (PDF)
├── models/ # 本地模型权重 (BGE)
├── templates/ # 前端界面模板
├── vectorstore/ # Chroma 向量数据库
├── .env # 环境变量
├── Dockerfile # Docker 构建文件
├── main.py # FastAPI 入口
├── requirements.txt # Python 依赖
└── README.md

text

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| 大模型 | 通义千问 (qwen-turbo) |
| 向量存储 | Chroma |
| 嵌入模型 | BGE-large-zh-v1.5 |
| 重排序 | BGE-reranker-base |
| 语义缓存 | FAISS |
| 评测框架 | Ragas |

## 📋 环境要求

- Python 3.10+
- 通义千问 API Key
- 建议 16GB+ 内存 (运行本地 BGE 模型)

## 🚀 快速开始

1. 克隆项目
```bash
git clone https://github.com/uu780/fraud-qa-assistant.git
cd fraud-qa-assistant
2.安装依赖
bash
pip install -r requirements.txt
3. 配置 API Key
bash
echo "TONGYI_API_KEY=你的API密钥" > .env
4. 构建知识库索引
bash
cd app
python index.py
cd ..
5. 启动服务
bash
cd app
python main.py
访问 http://localhost:8089 即可使用。

🐳 Docker 部署
构建镜像
bash
docker build -t fraud-qa-assistant .
运行容器
bash
docker run -d \
  --name fraud-qa \
  -p 8089:8089 \
  -e TONGYI_API_KEY="你的密钥" \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/vectorstore:/app/vectorstore \
  fraud-qa-assistant
通过 -v 挂载模型和索引文件夹，避免镜像体积过大并实现数据持久化。
## 免责与合规声明
> 本项目仅用于**公益反诈科普学习**，所有训练素材、文本资料均来源于网络公开合规内容。
> 项目代码开源分享，**原始数据集、向量库(vector store)均未上传**，请使用者自行通过正规官方渠道获取合规资料。
> 禁止用于商业用途、违规爬虫及违法场景，使用本项目产生的一切后果由使用者自行承担。
