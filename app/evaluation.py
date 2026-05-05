#evaluation.py:
import asyncio
import torch
import pandas as pd
from ragas import evaluate
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.metrics import ContextRelevance, answer_relevancy, faithfulness, ResponseGroundedness
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app.core import (
    get_retriever, get_bm25_retriever, get_llm,
    rephrase_retrieve, get_rag_chain
)

chat_history = []
retrieve_history=[]
embedding_model = HuggingFaceEmbeddings(
model_name=r"models\bge-large-zh-v1.5",
model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
encode_kwargs={
"normalize_embeddings": True
 },
)
llm = get_llm()
rerank_model = AutoModelForSequenceClassification.from_pretrained(r"models/bge-reranker-base")
rerank_tokenizer = AutoTokenizer.from_pretrained(r"models/bge-reranker-base")

async def invoke_rag(query,conversation_id):
    input = {"query": query, "history": chat_history}
    retriever = get_retriever(k=20, embedding_model=embedding_model)
    bm25_retriever = get_bm25_retriever()
    result= rephrase_retrieve(input,llm,bm25_retriever,retriever,rerank_tokenizer=rerank_tokenizer,rerank_model=rerank_model,query_num=5)
    rag_chain = get_rag_chain(result,llm)

    full_answer = ""
    async for chunk in rag_chain.astream({
        "query": input["query"],
        "history": input["history"]
    }):
        full_answer += chunk
        yield chunk
    chat_history.append(
        {"role": "user", "content": query, "conversation_id": conversation_id})
    chat_history.append(
        {"role": "ai", "content": full_answer, "conversation_id": conversation_id})
    retrieve_history.append({
        "query":query,
        "answer":full_answer,
        "contexts":[docs.page_content for docs in result]})

async def rag_evaluate(datas):
    ragas_data = {
                 "user_input": [d["query"] for d in datas],
                 "response": [d["answer"] for d in datas],
                 "retrieved_contexts": [d["contexts"] for d in datas]
    }
    dataset = Dataset.from_dict(ragas_data)
    metrics = [
        ContextRelevance(),
        answer_relevancy,
        faithfulness,
        ResponseGroundedness()
    ]
    result = evaluate(
        dataset = dataset,
        metrics = metrics,
        llm = llm,
        embeddings = embedding_model,
        raise_exceptions = False)
    datas.clear()
    return result

if __name__ == "__main__":
    async def main():
        query= "我给陌生人验证码会被诈骗吗"
        async for chunk in invoke_rag(query, 1):
            print(chunk, end="", flush=True)
        evaluate_res = await rag_evaluate(retrieve_history)
        print(evaluate_res)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(evaluate_res.to_pandas()[
        ["nv_context_relevance",
        "answer_relevancy",
        "faithfulness",
        "nv_response_groundedness"]])
    asyncio.run(main())