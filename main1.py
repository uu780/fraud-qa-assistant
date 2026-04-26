import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from core import (
    get_retriever, get_bm25_retriever, get_llm,
    rephrase_retrieve, get_rag_chain
)
from evaluation import (
    embedding_model, rerank_model, rerank_tokenizer
)

# 导入缓存管理器
from rag_pgm.cache.cache_manager import CacheManager

# ============================================================
# 初始化缓存
# ============================================================
# 使用语义缓存（FAISS）+ 内存精确匹配缓存
cache_manager = CacheManager(
    embedding_model=embedding_model,  # 复用你的 embedding_model
    semantic_threshold=0.85,  # 相似度阈值85%
    use_redis=False,  # 先用内存缓存，需要再改True
)

chat_history = []  # 对话历史


# ============================================================
# RAG核心函数（增加缓存逻辑）
# ============================================================
async def invoke_rag(query: str, conversation_id: int, history: list):
    """
    执行RAG查询，流式输出（带缓存）
    """

    # 1. 先检查缓存
    cached_response, cache_level = cache_manager.get(query)

    if cached_response:
        print(f"✅ 缓存命中！级别: {cache_level}, 查询: {query}")
        # 直接返回缓存的响应
        full_answer = cached_response
        yield full_answer  # 注意：缓存的是完整响应，不是流式的

        # 保存到历史
        history.append({"role": "user", "content": query, "conversation_id": conversation_id})
        history.append({"role": "ai", "content": full_answer, "conversation_id": conversation_id})
        return

    # 2. 缓存未命中，执行正常的RAG流程
    print(f"❌ 缓存未命中，执行RAG: {query}")

    # 构造输入
    input_data = {"query": query, "history": history}

    # 获取检索器和LLM
    retriever = get_retriever(k=20, embedding_model=embedding_model)
    bm25_retriever = get_bm25_retriever()
    llm = get_llm()

    # 执行检索
    result_docs = rephrase_retrieve(
        input_data,
        llm,
        retriever,
        bm25_retriever,
        rerank_tokenizer,
        rerank_model,
        query_num=5
    )

    # 获取RAG链
    rag_chain = get_rag_chain(result_docs, llm)

    # 流式生成回答
    full_answer = ""
    async for chunk in rag_chain.astream({
        "query": input_data["query"],
        "history": input_data["history"]
    }):
        full_answer += chunk
        yield chunk

    # 3. 保存到缓存
    cache_manager.set(query, full_answer)
    print(f"💾 已缓存: {query}")

    # 保存到历史
    history.append({"role": "user", "content": query, "conversation_id": conversation_id})
    history.append({"role": "ai", "content": full_answer, "conversation_id": conversation_id})


# ============================================================
# FastAPI应用
# ============================================================
app = FastAPI(title="防诈骗问答助手")

# 挂载静态文件
app.mount("/static", StaticFiles(directory="templates"), name="static")


@app.get("/")
async def homepage():
    """返回首页"""
    return FileResponse("templates/naive_index.html")


@app.get("/stream_response")
async def stream_response(query: str):
    """流式响应接口"""
    return StreamingResponse(
        invoke_rag(query, 1, chat_history),
        media_type="text/event-stream"
    )


@app.get("/history")
async def get_history():
    """获取对话历史"""
    return {"history": chat_history}


@app.delete("/history")
async def clear_history():
    """清空对话历史"""
    global chat_history
    chat_history = []
    return {"message": "历史已清空"}


# 新增：缓存管理接口
@app.get("/cache/stats")
async def get_cache_stats():
    """获取缓存统计信息"""
    return cache_manager.stats()


@app.delete("/cache")
async def clear_cache():
    """清空所有缓存"""
    cache_manager.clear()
    return {"message": "缓存已清空"}


@app.get("/cache/status")
async def cache_status():
    """检查缓存状态"""
    stats = cache_manager.stats()
    return {
        "status": "running",
        "semantic_cache_size": stats["semantic_cache"].get("size", 0),
        "response_cache_size": stats["response_cache"].get("size", 0),
    }


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8089)