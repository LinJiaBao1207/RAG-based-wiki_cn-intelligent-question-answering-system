import os
import json
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# 优先加载当前目录或上级的 .env 文件
load_dotenv()
# 回退加载特定路径
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# 强制重写环境变量以便找到模型
os.environ["LOCAL_EMBED_MODEL_PATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "bge-m3"))
os.environ["LOCAL_RERANK_MODEL_PATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "bge-reranker-v2-m3"))
os.environ["SENTENCEPIECE_MODEL_PATH"] = os.path.join(os.environ["LOCAL_EMBED_MODEL_PATH"], "sentencepiece.bpe.model")

# 引入系统现有的 RAG 模块前必须先设置好环境变量
from qa_web.app.rag_engine import RagEngine

# ================= 配置区 =================
# 从 .env 读取阿里云百炼的配置（通过兼容 OpenAI SDK 的方式调用）
EVAL_API_KEY = os.getenv("BAILIAN_API_KEY", "")
EVAL_BASE_URL = os.getenv("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
# 推荐使用 qwen-max 作为裁判模型
EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "qwen-max")
# 指定百炼的 embedding 模型
EVAL_EMBED_MODEL = os.getenv("EVAL_EMBED_MODEL", "text-embedding-v3")
# 自动生成 reference（与检索证据对齐）
AUTO_REFERENCE = os.getenv("AUTO_REFERENCE", "0") == "1"
# 生成模式：missing 仅在 reference 为空时生成；all 覆盖全部
AUTO_REFERENCE_MODE = os.getenv("AUTO_REFERENCE_MODE", "missing")

# 如果您有本地生成的测试集，可以修改路径，格式见代码中的示例。
TESTSET_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.jsonl")

def build_testset():
    """
    如果测试集不存在，生成一个简单的示例测试集。
    在实际使用中，你需要使用真实业务场景构建包含 ground_truth 的评测数据。
    """
    sample_data = [
        {
            "question": "Wiki-CN系统是由哪些组件构成的？",
            "ground_truth": "Wiki-CN 问答系统主要包含基于 BGE-M3 的检索与 Reranker 重排引擎，以及最终用于生成的 LLM 大模型（如 Qwen 或 GPT）。"
        },
        # 在这里添加更多的问题和对应的正确参考答案（Ground Truth）
    ]
    with open(TESTSET_PATH, "w", encoding="utf-8") as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已生成示例测试集: {TESTSET_PATH}")


_ref_llm = None


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _generate_reference(question: str, refs: list[dict]) -> str:
    global _ref_llm
    if not refs:
        return "证据不足"

    if _ref_llm is None:
        _ref_llm = ChatOpenAI(
            api_key=EVAL_API_KEY,
            base_url=EVAL_BASE_URL,
            model=EVAL_MODEL_NAME,
            temperature=0.0,
        )

    context_blocks = []
    for i, r in enumerate(refs[:5], start=1):
        context_blocks.append(
            f"[证据{i}] 标题：{r.get('title','')}\n"
            f"URL：{r.get('url','')}\n"
            f"片段：{str(r.get('full_text',''))[:800]}"
        )

    system_prompt = (
        "你是一个严格基于证据的评测助手。"
        "只允许使用证据中的事实作答。"
        "如果证据不足以回答问题，请只输出：证据不足。"
        "输出必须简洁、单段文本、无列表。"
    )
    user_prompt = (
        f"问题：{question}\n\n"
        f"证据：\n{chr(10).join(context_blocks)}\n\n"
        "请输出与证据完全一致的参考答案。"
    )

    resp = _ref_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    text = getattr(resp, "content", "") if resp else ""
    return str(text).strip() or "证据不足"


def _print_trace(idx: int, question: str, trace: dict, contexts: list[str], answer: str, reference: str) -> None:
    print("\n--- Trace %d ---" % idx)
    print("Q:", question)
    print(
        "Retrieval:",
        trace.get("retrieval_mode", ""),
        "bm25_hits=", trace.get("bm25_hits", 0),
        "vec_hits=", trace.get("vec_hits", 0),
        "vec_used=", trace.get("vec_used", False),
        "rerank_used=", trace.get("rerank_used", False),
    )
    print(
        "Rewrite:",
        "ON" if trace.get("rewrite_used", False) else "OFF",
        "queries=", trace.get("rewrite_queries", []),
    )
    print(
        "Web:",
        "ON" if trace.get("web_used", False) else "OFF",
        "provider=", trace.get("web_provider", "none"),
    )
    print("Gen:", trace.get("gen_provider", ""), "model=", trace.get("gen_model", ""))
    print("Eval:", "model=", EVAL_MODEL_NAME, "embed=", EVAL_EMBED_MODEL)
    print("Contexts:", len(contexts))
    print("Answer:", answer)
    print("Reference:", reference)

def run_evaluation():
    if not os.path.exists(TESTSET_PATH):
        build_testset()

    print("正在初始化 RAG 系统...")
    engine = RagEngine()
    if AUTO_REFERENCE and not EVAL_API_KEY:
        print("【错误】AUTO_REFERENCE=1 但未检测到 BAILIAN_API_KEY。请检查 .env 文件。")
        return

    eval_data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
        "retrieval_mode": [],
        "bm25_hits": [],
        "vec_hits": [],
        "vec_used": [],
        "rerank_used": [],
        "web_used": [],
        "web_provider": [],
        "rewrite_used": [],
        "rewrite_queries": [],
        "gen_provider": [],
        "gen_model": [],
        "eval_model": [],
        "eval_embed_model": [],
        "llm_primary_provider": [],
        "dense_backend": []
    }

    print("开始回答测试集问题并收集上下文...")
    print(f"自动生成 reference: {'ON' if AUTO_REFERENCE else 'OFF'}")
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            question = _safe_text(item.get("question", "")).strip()
            ground_truth = _safe_text(item.get("ground_truth", "")).strip()
            if not question:
                continue
            
            # 先检索，再生成答案，保留 full_text 以便自动生成 reference
            answer, refs_full, refs = engine.ask_with_full_refs(question)
            answer = _safe_text(answer).strip()
            trace = engine.get_last_trace()
            bm25_hits = int(trace.get("bm25_hits", 0))
            vec_hits = int(trace.get("vec_hits", 0))
            vec_used = bool(trace.get("vec_used", False))
            rerank_used = bool(trace.get("rerank_used", False))
            if vec_used and bm25_hits > 0:
                retrieval_mode = "hybrid"
            elif vec_used:
                retrieval_mode = "dense"
            else:
                retrieval_mode = "bm25"
            trace["retrieval_mode"] = retrieval_mode
            
            # 提取文本片段用于 Ragas 评测 (检索到的原始文本)
            contexts = [_safe_text(ref.get("snippet", "")).strip() for ref in refs]
            contexts = [c for c in contexts if c]
            if AUTO_REFERENCE and (AUTO_REFERENCE_MODE == "all" or not ground_truth):
                ground_truth = _generate_reference(question, refs_full)
            ground_truth = _safe_text(ground_truth).strip()
            if not ground_truth:
                ground_truth = "证据不足"

            _print_trace(
                idx=len(eval_data["user_input"]) + 1,
                question=question,
                trace=trace,
                contexts=contexts,
                answer=answer,
                reference=ground_truth,
            )
            
            # 现在的 Ragas API 期望这两种列是类似数组格式或者特定的 user_input/reference
            eval_data["user_input"].append(question)
            eval_data["response"].append(answer)
            eval_data["retrieved_contexts"].append(contexts)
            eval_data["reference"].append(ground_truth)
            eval_data["retrieval_mode"].append(retrieval_mode)
            eval_data["bm25_hits"].append(bm25_hits)
            eval_data["vec_hits"].append(vec_hits)
            eval_data["vec_used"].append(vec_used)
            eval_data["rerank_used"].append(rerank_used)
            eval_data["web_used"].append(bool(trace.get("web_used", False)))
            eval_data["web_provider"].append(str(trace.get("web_provider", "none")))
            eval_data["rewrite_used"].append(bool(trace.get("rewrite_used", False)))
            eval_data["rewrite_queries"].append(trace.get("rewrite_queries", []))
            eval_data["gen_provider"].append(str(trace.get("gen_provider", "")))
            eval_data["gen_model"].append(str(trace.get("gen_model", "")))
            eval_data["eval_model"].append(EVAL_MODEL_NAME)
            eval_data["eval_embed_model"].append(EVAL_EMBED_MODEL)
            eval_data["llm_primary_provider"].append(str(trace.get("llm_primary_provider", "")))
            eval_data["dense_backend"].append(str(trace.get("dense_backend", "")))

    dataset = Dataset.from_dict(eval_data)

    print("初始化 RAGAS 裁判大模型 (基于阿里云百炼)...")
    if not EVAL_API_KEY:
        print("【错误】未检测到 BAILIAN_API_KEY。请检查 .env 文件。")
        return

    # 用于作为裁判的大模型，通过兼容 OpenAI 的接口请求百炼
    evaluator_llm = ChatOpenAI(
        api_key=EVAL_API_KEY,
        base_url=EVAL_BASE_URL,
        model=EVAL_MODEL_NAME,
        temperature=0.0
    )
    # 用于计算向量相似度相关指标的 Embedding模型（百炼 Embedding）
    evaluator_embeddings = OpenAIEmbeddings(
        api_key=EVAL_API_KEY,
        base_url=EVAL_BASE_URL,
        model=EVAL_EMBED_MODEL
    )

    ragas_llm = LangchainLLMWrapper(evaluator_llm)
    # 对于 Ragas 0.2+ 版本，我们不再传入 embeddingsWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    ragas_embeddings = LangchainEmbeddingsWrapper(evaluator_embeddings)
    
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    
    m_faithfulness = Faithfulness()
    m_answer_relevance = AnswerRelevancy()
    m_context_precision = ContextPrecision()
    m_context_recall = ContextRecall()

    # 将裁判模型和向量模型注入到 Ragas 指标中
    for metric in [m_faithfulness, m_answer_relevance, m_context_precision, m_context_recall]:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    print("开始执行 RAGAS 自动化评估 (这可能需要几分钟时间)...")
    result = evaluate(
        dataset,
        metrics=[
            m_context_precision, # 检索指标：有用信息是否排在最前面
            m_context_recall,    # 检索指标：能够回答问题的上下文是否都被召回
            m_faithfulness,      # 生成指标：答案是否忠实于原文（幻觉评估）
            m_answer_relevance,  # 生成指标：答案是否直接回答了问题
        ],
    )

    print("\n========== 评估结果 ==========")
    print(result)
    
    # 导出详细的带分数的评测结果，用于深度分析 Bad Case
    df_eval = pd.DataFrame(eval_data)
    df_scores = result.to_pandas()
    metric_cols = [c for c in df_scores.columns if c not in df_eval.columns]
    df_out = pd.concat([df_eval, df_scores[metric_cols]], axis=1)
    df_out.to_csv("ragas_evaluation_report.csv", index=False, encoding="utf-8-sig")
    print("详细报告已保存至: ragas_evaluation_report.csv")

if __name__ == "__main__":
    run_evaluation()
