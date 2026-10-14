import argparse
import os
import sys
import types
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ragas 0.4.x imports ChatVertexAI from langchain_community.chat_models.vertexai,
# which was removed in langchain-community 0.4. Inject a shim before ragas loads.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI

    _shim = types.ModuleType("langchain_community.chat_models.vertexai")
    _shim.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _shim


from google import genai
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools.retriever import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.metrics._faithfulness import Faithfulness

# from ragas.llms import llm_factory


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------

CHUNK_SIZE = 256


def load_codebase(repo_path: str) -> list:
    docs = []
    for path in Path(repo_path).rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"source": str(path)}))
    return docs


# ---------------------------------------------------------------------------
# 2. CHUNK
# ---------------------------------------------------------------------------


def chunk_code(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=32,
    )
    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# 3. EMBED & STORE
# ---------------------------------------------------------------------------


def build_vector_store(chunks: list) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embedding=embeddings)


# ---------------------------------------------------------------------------
# 4. RETRIEVER + AGENT
# ---------------------------------------------------------------------------


def build_agent(vector_store: Chroma):
    retriever_tool = create_retriever_tool(
        vector_store.as_retriever(search_kwargs={"k": 4}),
        name="search_codebase",
        description="Search the codebase for relevant functions, classes, or logic.",
    )

    #    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    #    evaluator_llm = llm_factory(
    #         model="gemini-2.5-flash",
    #         provider="google",
    #         client=client,
    #         temperature=0,
    #     )

    evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    return create_agent(
        evaluator_llm,
        tools=[retriever_tool],
        system_prompt=(
            "You are a senior engineer. Always use search_codebase before answering. "
            "Reference specific file and function names. "
            "If not found say 'I could not find that in the codebase'."
        ),
        middleware=[
            ModelCallLimitMiddleware(run_limit=5, exit_behavior="end"),
            ToolCallLimitMiddleware(
                tool_name="retriever_tool", run_limit=2, exit_behavior="end"
            ),
        ],
    )


def run_agent(agent, question: str) -> tuple[str, list[str]]:
    """Invoke the agent and extract (answer, contexts) from the message trace."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].content

    contexts = []
    for msg in result["messages"]:
        if type(msg).__name__ == "ToolMessage" and isinstance(msg.content, str):
            contexts.append(msg.content)

    return answer, contexts


# ---------------------------------------------------------------------------
# Golden Dataset - Ground-truth test set (questions about sample_project)
# ---------------------------------------------------------------------------

from golden_dataset import TEST_CASES

# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default=str(Path(__file__).parent.parent / "sample_project"),
    )
    args = parser.parse_args()
    repo_path = str(Path(args.repo).resolve())

    docs = load_codebase(repo_path)
    chunks = chunk_code(docs)
    print(f"Loaded {len(docs)} files -> {len(chunks)} chunks (chunk_size={CHUNK_SIZE})")

    vector_store = build_vector_store(chunks)
    agent = build_agent(vector_store)

    # Collect RAG outputs
    print("\nRunning agent on test cases...")
    samples = []
    for test_case in TEST_CASES:
        answer, contexts = run_agent(agent, test_case["question"])

        if isinstance(answer, list):
            answer = "".join(
                block.get("text", "")
                for block in answer
                if isinstance(block, dict) and block.get("type") == "text"
            )

        samples.append(
            SingleTurnSample(
                user_input=test_case["question"],
                response=answer,
                retrieved_contexts=contexts,
                reference=test_case["reference"],
            )
        )

        print(f"\nQ: {test_case['question']}")
        print(f"Contexts retrieved ({len(contexts)}):")
        for i, ctx in enumerate(contexts, 1):
            print(f"  [{i}] {ctx[:200]}{'...' if len(ctx) > 200 else ''}")
        print(f"Answer: {answer}")

    # RAGAS evaluation
    print("\nRunning RAGAS evaluation (gemini-2.5-flash as judge)...")
    # evaluator_llm = llm_factory(model="gemini-2.5-flash", provider="google", temperature=0)

    evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    lc_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    eval_dataset = EvaluationDataset(samples=samples)

    results = evaluate(
        dataset=eval_dataset,
        metrics=[
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            Faithfulness(),
            AnswerRelevancy(embeddings=lc_embeddings),
            FactualCorrectness(),
        ],
        llm=evaluator_llm,
    )

    # Scorecard
    df = results.to_pandas()
    metric_cols = [
        "llm_context_precision_with_reference",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
        "factual_correctness",
    ]

    print("\nRAGAS SCORECARD - Semantic RAG on sample_project")
    print("\nPer-question breakdown:")
    for _, row in df.iterrows():
        print(f"\n  Q: {row['user_input'][:65]}")
        for col in metric_cols:
            if col in row:
                val = row[col]
                icon = "✅" if val >= 0.7 else ("⚠️ " if val >= 0.5 else "❌")
                print(f"    {icon} {col:<42}: {val:.3f}")

    print("\nAggregate averages:")
    for col in metric_cols:
        if col in df.columns:
            avg = df[col].mean()
            icon = "✅" if avg >= 0.7 else ("⚠️ " if avg >= 0.5 else "❌")
            print(f"  {icon} {col:<42}: {avg:.3f}")
    print()
