import os
import jieba
import torch
from hashlib import sha256
from typing import Dict
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.llms.tongyi import Tongyi
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

def format_history(history, max_epoch=3):
    if len(history) > 2 * max_epoch:
        history = history[-2 * max_epoch:]
    return "\n".join([f"{i['role']}：{i['content']}" for i in history])

def format_docs(docs:list[Document]) -> str:
    """拼接多个 Document 的 page_content """
    return "\n\n".join(doc.page_content for doc in docs)

#检索--查询嵌入
def get_retriever(k=20,embedding_model=None):
    vectorstore = Chroma(persist_directory = "vectorstore",embedding_function = embedding_model)
    retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": k})
    return retriever
#关键词检索
def get_bm25_retriever(k=20):
    vectorstore = Chroma(persist_directory="vectorstore")
    all_docs = vectorstore.get()
    bm25_retriever = BM25Retriever.from_documents(
        [
            Document(id=doc_id, page_content=doc, metadata=meta)
            for doc_id, doc, meta in zip(
            all_docs["ids"], all_docs["documents"], all_docs["metadatas"]
        )
        ],
        preprocess_func=lambda x: jieba.lcut(x),
    )
    bm25_retriever.k = k
    return bm25_retriever

def get_llm():
    load_dotenv()
    TONGYI_API_KEY= os.getenv("TONGYI_API_KEY")
    llm = Tongyi(model="qwen-turbo",api_key=TONGYI_API_KEY)
    return llm


#重排序+重排分
def reciprocal_rank_fusion_with_rerank(docs: list[list[Document]],
                                       query: str,
                                       rerank_tokenizer,
                                       rerank_model,
                                       k=60,
                                       docs_return_num=10,
                                       rerank_top_n=30
                                       )-> list[Document]:
    fused_scores = {}
    unique_docs_by_content={}
    for doc_list in docs:
        for rank, doc in enumerate(doc_list):
            key = sha256(doc.page_content.encode("utf-8")).hexdigest()
            unique_docs_by_content[key] = doc
            fused_scores[key]= fused_scores.get(key,0)+ 1 / (rank + k)
    sorted_content_hashes = sorted(fused_scores.items(), key = lambda x: x[1], reverse = True)
    top_candidates = sorted_content_hashes[:rerank_top_n]
    reranked_docs = []
    for i, _ in top_candidates:
        reranked_docs.append(unique_docs_by_content[i])
    inputs = rerank_tokenizer(
    text = [query] * len(reranked_docs),
    text_pair = [doc.page_content for doc in reranked_docs],
    padding = True,
    max_length = rerank_tokenizer.model_max_length,
    truncation = True,
    return_tensors = "pt")
    scores=[]
    with torch.no_grad():
        outputs = rerank_model(**inputs)
        batch_scores = outputs.logits.squeeze()
    scores.extend(batch_scores.tolist())
    final_docs=[doc for doc, _ in sorted(zip(reranked_docs, scores), key=lambda x: x[1], reverse=True)]
    print(f"Rerank scores 前10: {scores[:10]}")
    print(f"最终返回数量: {len(final_docs)}")
    return final_docs[:docs_return_num]


#检索-相似度检索--重述问题+多查询+提取上下文
def rephrase_retrieve(input: Dict[str, str],llm,retriever,bm25_retriever,rerank_tokenizer,rerank_model,query_num):
    rephrase_prompt=PromptTemplate.from_template(
       """
       根据对话历史简要完善最新的用户消息，使其更加具体。只输出完善后的问题。如果问题不需要完善，请直接输出原始问题。
       当前对话:{history}
       用户问题：{query}
       """)
    rephrase_chain=(({"query":lambda x:x["query"],"history": lambda x:x["history"]})
                     |rephrase_prompt
                     |llm
                     |StrOutputParser())
    rephrased_query = rephrase_chain.invoke({
        "query": input.get("query"),
        "history": input.get("history")
    })
    multi_query_prompt=PromptTemplate.from_template(
        """
        你是一名AI语言模型助理。你的任务是生成给定问题的{query_num}个不同版本，以从矢量数据库中检索相关文档。
        你需要通过从多个视角生成问题，来克服基于距离的相似性搜索的一些局限性。请使用换行符分隔备选问题。           
        原始问题：{query}
         """)
    multi_query_chain = (
            {"query": lambda x: x, "query_num": lambda x: query_num}
            | multi_query_prompt
            | llm
            | StrOutputParser()
            | (lambda x: x.strip().split("\n"))
    )
    queries = multi_query_chain.invoke(rephrased_query)
    vector_results = []
    for q in queries:
        results = retriever.invoke(q)
        vector_results.extend(results)
    bm25_results = []
    for q in queries:
        results = bm25_retriever.invoke(q)
        bm25_results.extend(results)
    result= reciprocal_rank_fusion_with_rerank([vector_results,bm25_results],
                                               query=input.get("query"),
                                               rerank_tokenizer=rerank_tokenizer,
                                               rerank_model=rerank_model,
                                               k=60,
                                               docs_return_num=10,
                                               rerank_top_n=30)
    return result

#生成answer
def get_rag_chain(result,llm):
    prompt = PromptTemplate(
    input_variables = ["context", "history", "query"],
    template =
       """
       你是一个专业的中文问答助手，擅长基于提供的资料回答问题。
       请仅根据以下背景资料以及历史消息回答问题，如无法找到答案，请直接回答“我不知道”。
       背景资料：{context}
       历史消息：[{history}]
       问题：{query}
       回答：
       """)
    rag_chain=({"context":lambda x: format_docs(result),"history":lambda x: format_history(x.get("history")),"query": lambda x: x.get("query")}
                |prompt
                |llm
                |StrOutputParser())
    return rag_chain


# ============================================================
# 测试代码（直接运行 core.py 时执行）
# ============================================================
#if __name__ == "__main__":
    from evaluation import embedding_model, rerank_model, rerank_tokenizer

    print("=" * 50)
    print("开始测试 RAG 检索流程")
    print("=" * 50)

    # 1. 初始化组件
    print("\n1. 初始化检索器...")
    retriever = get_retriever(k=20, embedding_model=embedding_model)
    bm25_retriever = get_bm25_retriever(k=20)
    llm = get_llm()
    print("   ✅ 初始化完成")

    # 2. 构造测试输入
    test_input = {
        "query": "创投精英群是什么骗局？",
        "history": []  # 空历史
    }

    print(f"\n2. 测试查询: {test_input['query']}")

    # 3. 执行检索
    print("\n3. 执行多查询扩展 + 混合检索 + RRF融合 + Reranker...")
    result_docs = rephrase_retrieve(
        input=test_input,
        llm=llm,
        retriever=retriever,
        bm25_retriever=bm25_retriever,
        rerank_tokenizer=rerank_tokenizer,
        rerank_model=rerank_model,
        query_num=3  # 生成3个查询变体
    )

    # 4. 打印结果
    print("\n" + "=" * 50)
    print(f"最终返回文档数量: {len(result_docs)}")
    print("=" * 50)

    for i, doc in enumerate(result_docs):
        print(f"\n--- 文档 {i + 1} ---")
        print(f"来源: {doc.metadata.get('source', '未知')}")
        print(f"内容预览: {doc.page_content[:200]}...")
        print(f"内容长度: {len(doc.page_content)} 字符")

    print("\n" + "=" * 50)
    print("测试完成")