import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel(
    'BAAI/bge-m3',
    device="gpu"
)

class BGEM3Embedding:
    def embed_documents(self, texts):
        if not texts:
            return []

        return model.encode(
            texts,
            batch_size=8,
            max_length=512,
            return_dense=True
        )["dense_vecs"].tolist()

    def embed_query(self, text):
        return model.encode(
            [text],
            batch_size=1,
            max_length=512,
            return_dense=True
        )["dense_vecs"][0].tolist()


def get_embeddings():
    return BGEM3Embedding()