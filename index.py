import re
import json
import torch
from transformers import AutoTokenizer
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_documents():
    pdf_documents = PyMuPDFLoader(r"C:\Users\Lenovo\PyCharmMiscProject\rag_pgm\data\raw\PDF合并.pdf").load()
    print ([n.page_content for n in pdf_documents] )
    return pdf_documents

def llm_repair_text(broken_text: str, llm) -> str:
    """
    用 LLM 修复常见文本问题：异常空格、OCR错别字、错误断句
    """
    prompt = f"""
你是一个文本修复专家。请修复以下文档中的问题：

1. 合并被错误换行或空格断开的句子（如“不 光” → “不光”）
2. 修正明显的 OCR 错别字（如“己经” → “已经”）
3. 删除多余空格

只输出修复后的文本，不要加解释，不要加额外内容。

待修复文本：
{broken_text}
"""
    response = llm.invoke(prompt)
    return response.content.strip()

def clean_documents(documents: list,use_llm_repair=False, llm=None):  #索引--文档清洗
    cleaned_docs = []
    for doc in documents:
        text = doc.page_content
        # 1. 去除金山文档/微信链接等无关信息
        text = re.sub(r"https?://\S+", "", text)  # 去除链接
        text = re.sub(r"收藏时间：\d{4}年\d{2}月\d{2}日", "", text)  # 去除时间
        text = re.sub(r"本文档由.*一键生成", "", text)  # 去除生成信息
        text = re.sub(r"📌 原文链接：.*?📑[\'\"\s]*", "", text, flags=re.DOTALL)

        # 2. 去除特殊转义序列
        text = re.sub(r"\\u[0-9a-fA-F]{4}", "", text)  # \uf359 等
        text = re.sub(r"\\x[0-9a-fA-F]{2}", "", text)  # \x1f 等
        # 3. 去除页码标记（⸺数字⸺ 格式）
        text = re.sub(r"⸺\d+⸺", "", text)

        # 4. 去除重复的页眉页脚（保留一次）
        text = re.sub(r"(来源：央视新闻\s*)+", "来源：央视新闻\n", text)
        text = re.sub(r"(公安部刑侦局\s*)+", "公安部刑侦局\n", text)

        # 5. 去除孤立的数字/符号行
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        # 2. 合并多余换行符，还原正常段落
        text = re.sub(r"\n+", "\n", text)  # 多个换行合并成一个
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # 段落内的换行换成空格
        text = re.sub(r"\n{2,}", "\n\n", text)  # 段落间保留一个空行
        # 3. 去除多余空格和首尾空白
        text = re.sub(r" +", " ", text).strip()
        if use_llm_repair and llm:
            text = llm_repair_text(text, llm)
        doc.page_content = text
        for key, value in doc.metadata.items():
            if not isinstance(value, (str, int, float, bool)):
                try:
                    doc.metadata[key] = json.dumps(value, ensure_ascii=False)
                except (TypeError, ValueError):
                    doc.metadata[key] = str(value)
        cleaned_docs.append(doc)
    print([d.page_content for d in cleaned_docs])
    return cleaned_docs


tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\Lenovo\PyCharmMiscProject\rag_pgm\app\models\bge-large-zh-v1.5")
def bge_token_len(text: str) -> int:
    return len(tokenizer.encode(text))

def split_by_content_type(documents):
    """根据内容类型选择分块策略"""
    keyword_chunks = []  # 关键词类（不分块）
    case_chunks = []  # 案例类（按语义切分）

    for doc in documents:
        content = doc.page_content

        # 判断是否为「防诈关键词」
        if (re.search(r'## \d+\.', content) and "名词解释" in content) or "二十个防诈关键词" in content:
            # 关键词类：整个作为独立 chunk
            doc.metadata["doc_type"] = "keyword"
            keyword_chunks.append(doc)
        else:
            # 案例类：按语义切分
            doc.metadata["doc_type"] = "case"
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
                chunk_size=350,
                chunk_overlap=35,
                length_function=bge_token_len
            )
            case_chunks.extend(splitter.split_documents([doc]))
    chunked_docs = keyword_chunks + case_chunks
    return chunked_docs
def save_to_db(chunks):
#索引--向量嵌入
    print("=== 进入 save_to_db 函数 ===")  # ✅ 加这
    print(f"chunks 数量: {len(chunks)}")
    embedding_model = HuggingFaceEmbeddings(model=r"C:\Users\Lenovo\PyCharmMiscProject\rag_pgm\app\models\bge-large-zh-v1.5",
                                            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                                            encode_kwargs={"normalize_embeddings": True})
    #索引--向量存储
    vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory="vectorstore")
    return vectorstore
# 测试检索
    #query = "创投精英群"
    #results = vectorstore.similarity_search(query, k=20)

    #print(f"检索结果数: {len(results)}")
    #for i, doc in enumerate(results):
        #print(f"\n--- 结果 {i+1} ---")
        #print(doc.page_content[:300])
    #return results

if __name__ == "__main__":
    documents = load_documents()
    print(f"1. 原始文档数: {len(documents)}")

    cleaned_docs = clean_documents(documents)
    print(f"2. 清洗后文档数: {len(cleaned_docs)}")

    #from langchain_community.llms.tongyi import Tongyi
    # llm = Tongyi(model="qwen-turbo", api_key=os.getenv("TONGYI_API_KEY"))
    # cleaned_docs = clean_documents(documents, use_llm_repair=True, llm=llm)
    
    chunked_docs = split_by_content_type(cleaned_docs)
    print(f"3. 切分后 chunk 数: {len(chunked_docs)}")

    vectorstore = save_to_db(chunked_docs)

