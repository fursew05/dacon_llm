import os
from qdrant_client import QdrantClient, models
from qdrant_client.http import models
# from fastembed import TextEmbedding
from FlagEmbedding import BGEM3FlagModel
from dotenv import load_dotenv

load_dotenv()

# Qdrant 클라이언트 초기화 (로컬에서 실행 중인 Qdrant에 연결)
client = QdrantClient(host="211.47.56.70", port=6389)

# 벡터 생성을 위한 임베딩 모델 초기화
embedder_name = os.getenv('EMBEDDING_MODEL_NAME', None)
# embedding_model = TextEmbedding(embedder_name)
embedding_model = BGEM3FlagModel(embedder_name)

# Documents
import joblib
from tqdm import tqdm
documents = joblib.load('law_v3b_original_sentences.joblib')

# 컬렉션 생성
collection_name = "fc_collection_v3b"

client.create_collection(
    collection_name=collection_name,
    vectors_config={
        # 밀집 벡터 설정
        "dense": models.VectorParams(
            size=1024, 
            distance=models.Distance.COSINE,
        )
    },
    # 희소 벡터 설정
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(
            index=models.SparseIndexParams(
                on_disk=False,
            )
        )
    },
)
# 샘플 데이터
# documents = [
#     "Qdrant is a vector database & vector similarity search engine.",
#     "It deploys as an API service and is ready for production.",
#     "Hybrid search combines keyword-based and semantic search.",
#     "Fastembed is a lightweight library for generating embeddings.",
#     "This tutorial explains how to perform hybrid search with Qdrant.",
# ]



# 문서에 대해 벡터 생성 및 업로드
points_to_upload = []
for i, doc in enumerate(tqdm(documents)):
    # 밀집 벡터와 희소 벡터 생성
    embeddings = embedding_model.encode(doc, return_dense=True, return_sparse=True)
    
    dense_vector = embeddings['dense_vecs']
    sparse_vector = embeddings['lexical_weights']

    points_to_upload.append(
        models.PointStruct(
            id=i,
            payload={"text": doc},
            vector={
                "dense": dense_vector,
                "sparse": models.SparseVector(
                    indices=list(sparse_vector.keys()),
                    values=list(sparse_vector.values())
                ),
            },
        )
    )

# Qdrant에 데이터 업로드
client.upload_points(
    collection_name=collection_name,
    points=points_to_upload,
    wait=True,
)

print("🎉 데이터 인덱싱 완료!")