# # Import

from collections import defaultdict
import time, datetime
import re
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['HF_HOME'] = '~/.cache/huggingface'
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.reasoning import DeepSeekR1ReasoningParser, Qwen3ReasoningParser
from transformers import AutoTokenizer
from dotenv import load_dotenv
if os.getenv('MODEL_NAME') is None:
    load_dotenv()

# Embedding model imports
import numpy as np
from FlagEmbedding import BGEM3FlagModel, FlagReranker

from langchain_huggingface import HuggingFaceEmbeddings

from qdrant_client import QdrantClient, models

# 객관식 여부 판단 함수
def is_multiple_choice(question_text):
    """
    객관식 여부를 판단: 2개 이상의 숫자 선택지가 줄 단위로 존재할 경우 객관식으로 간주
    """
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
    return option_count >= 2


# 질문과 선택지 분리 함수
def extract_question_and_choices(full_text):
    """
    전체 질문 문자열에서 질문 본문과 선택지 리스트를 분리
    """
    lines = full_text.strip().split("\n")
    q_lines = []
    options = []

    for line in lines:
        if re.match(r"^\s*[1-9][0-9]?\s", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())
    
    question = " ".join(q_lines)
    return question, options

# 프롬프트 생성기
def make_prompt_context(text, context):
    context_formatted = '\n\n'.join(context).strip()
    
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        system_prompt = f"""You are an expert at financial security and a test-taker.
Select the correct answer label (number) among <options> for the following <problem>.
You may refer to the given <context> if necessary.
Provide only the answer label number and do not add extra explanations."""

        prompt = f"""
<context>
{context_formatted}
</context>

<problem>
{question}
</problem>

<options>
{chr(10).join(options)}
</options>

Your answer:"""

    else:
        system_prompt = f"""You are an expert at financial security and a test-taker.
Write an accurate and brief answer for the following <problem>.
You may refer to the given <context> if necessary.
Beware that the answer must be in natural and fluent Korean. Provide only the answer in short sentences. Do not add extra response other than the answer. Do not provide answer as Markdown format. Provide answer in a plain text."""

        prompt = f"""
<context>
{context_formatted}
</context>

<problem>
{text}
</problem>

Your answer:"""

    return system_prompt, prompt

def inference_vllm(prompt, system_prompt, model, parser=None, max_new_tokens=8192, temperature=0.6, top_p=0.95, **kwargs):
    params = SamplingParams(temperature=temperature,
                           top_p=top_p,
                           max_tokens=max_new_tokens,
                           **kwargs)

    messages = [
        {
            'role': 'system',
            'content': system_prompt,
        },
        {
            'role': 'user',
            'content': prompt
        }
    ]

    output = model.chat(messages, params, use_tqdm=False)

    text = output[0].outputs[0].text
    if parser:
        reasoning_content, content = parser.extract_reasoning_content(text, None)
    else:
        content = text
    return content

def extract_answer_only(generated_text: str, original_question: str) -> str:
    # "답변:" 기준으로 텍스트 분리
    if "Your answer:" in generated_text:
        text = generated_text.split("Your answer:")[-1].strip()
    else:
        text = generated_text.strip()
    
    # 공백 또는 빈 문자열일 경우 기본값 지정
    if not text:
        return "미응답"

    # 객관식 여부 판단
    is_mc = is_multiple_choice(original_question)

    if is_mc:
        # 숫자만 추출
        match = re.match(r"\D*([1-9][0-9]?)", text)
        if match:
            return match.group(1)
        else:
            # 숫자 추출 실패 시 "0" 반환
            return "0"
    else:
        return text

def hybrid_search(query, embedder, qdrant_client: QdrantClient, collection_name, top_k, fusion_method='rrf', weights=[0.8, 0.2]):

    # 쿼리에 대한 벡터 생성
    query_embeddings = embedder.encode(query, return_dense=True, return_sparse=True)
    query_dense = query_embeddings['dense_vecs']
    query_sparse_dict = query_embeddings['lexical_weights']

    # 하이브리드 검색 실행
    search_results = qdrant_client.query_points(
        collection_name=collection_name,
        query=models.FusionQuery(
            fusion=fusion_method
        ),
        prefetch=[
            # Dense vector
            models.Prefetch(
                query=query_dense,
                using='dense',
                limit=top_k
            ),
            # Sparse vector
            models.Prefetch(
                query=models.SparseVector(
                    indices=list(query_sparse_dict.keys()),
                    values=list(query_sparse_dict.values())
                ),
                using='sparse',
                limit=top_k
            )

        ]
    ).points

    top_k_chunks = [point.payload['text'] for point in search_results]
    return top_k_chunks

def rerank(query, chunks, reranker, top_n):
    if not chunks:
        return []
        
    # Create pairs of [query, chunk] for scoring
    sentence_pairs = [[query, chunk] for chunk in chunks]
    
    # Compute the relevance scores
    scores = reranker.compute_score(sentence_pairs)
    
    # Combine chunks with their scores
    scored_chunks = list(zip(chunks, scores))
    
    # Sort chunks by score in descending order
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top_n reranked chunks (text only)
    reranked_chunks = [chunk for chunk, score in scored_chunks[:top_n]]
    
    return reranked_chunks

def main():
    tag = os.getenv("EXPR_TAG")  # 
    print(tag)

    model_name = os.getenv('MODEL_NAME')
    print(model_name)
    
    # LLM
    s_time = time.perf_counter()
    model = LLM(model=model_name, 
                # quantization='awq',
                # dtype="bfloat16", 
                enable_prefix_caching=True)
    print('🕑 LLM loading took', time.perf_counter() - s_time)

    # Embedding model
    print('Loading embedding models', os.getenv('EMBEDDING_MODEL_NAME'))
    s_time = time.perf_counter()
    embedder_name = os.getenv('EMBEDDING_MODEL_NAME')
    if 'bge-m3' in embedder_name.lower():
        embedder = BGEM3FlagModel(os.getenv('EMBEDDING_MODEL_NAME'), use_fp16=True)
    else:
        embedder = HuggingFaceEmbeddings(model_name=os.getenv('EMBEDDING_MODEL_NAME'),
                                        model_kwargs={
                                            #  'torch_dtype': True,
                                            'device': 'cuda'
                                        },
                                        encode_kwargs={
                                            'normalize_embeddings': True
                                        })

    # RAG index
    qdrant_client = QdrantClient(host="211.47.56.70", port=6389)
    collection_name = 'fc_collection_v3b'
    
    top_k = 20

    # Retrievers
    fusion_method = 'rrf'
    retriever_weights = [0.8, 0.2]
    
    # Reranker
    # top_n = 20
    # reranker = FlagReranker('dragonkue/bge-reranker-v2-m3-ko')
    
    # Output parser
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if 'deepseek' in model_name.lower():
        parser = DeepSeekR1ReasoningParser(tokenizer)
    elif 'qwen' in model_name.lower():
        parser = Qwen3ReasoningParser(tokenizer)
    else:
        parser = None
    
    test = pd.read_csv('./test.csv')
    preds = []
    retrieved_chunks = []

    avg_inf_time = 0
    for q in tqdm(test['Question'], desc="Inference"):
        relevant_chunks = hybrid_search(q, embedder, qdrant_client, collection_name, top_k, fusion_method, retriever_weights)
        # reranked_chunks = rerank(q, relevant_chunks, reranker, top_n)

        retrieved_chunks.append(relevant_chunks)
        system_prompt, prompt = make_prompt_context(q, relevant_chunks)

        s_time = time.perf_counter()
        output = inference_vllm(prompt, system_prompt, model, parser,
                                temperature=0.6,
                                top_p=0.95,
                                top_k=20
                                )
        inf_time = time.perf_counter() - s_time
        print(output.strip())
        print(inf_time, 'seconds took.')
        avg_inf_time += inf_time
        
        output = output.replace('**', '').strip()

        pred_answer = extract_answer_only(output, original_question=q)
        preds.append(pred_answer)

    print('Average inference time in seconds:', avg_inf_time / len(test['Question']))

    sample_submission = pd.read_csv('./sample_submission.csv')
    sample_submission['Answer'] = preds
    sample_submission['Retrieved Chunks'] = retrieved_chunks
    
    today = datetime.datetime.now().strftime('%m%d')
    sample_submission.to_csv(f'./{today}_hybrid_search_{model_name.split("/")[-1]}_{tag}.csv', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    s_time = time.perf_counter()
    main()
    print('🕑 Total elapsed time:', time.perf_counter() - s_time)
