import functools
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


@functools.lru_cache(maxsize=1)
def _get_model():
    from FlagEmbedding import BGEM3FlagModel
    return BGEM3FlagModel("BAAI/bge-m3", device="cpu")


class BGEM3Embedding:
    def embed_documents(self, texts):
        if not texts:
            return []

        return _get_model().encode(
            texts,
            batch_size=8,
            max_length=512,
            return_dense=True
        )["dense_vecs"].tolist()

    def embed_query(self, text):
        return _get_model().encode(
            [text],
            batch_size=1,
            max_length=512,
            return_dense=True
        )["dense_vecs"][0].tolist()


def get_embeddings():
    return BGEM3Embedding()
