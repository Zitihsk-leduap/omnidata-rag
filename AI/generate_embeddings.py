from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)


class BGEM3Embedding:
    def embed_documents(self, texts):
        return model.encode(
            texts,
            batch_size=4,
            max_length=512,
            return_dense=True
        )["dense_vecs"].tolist()

    def embed_query(self, text):
        return model.encode(
            [text],
            max_length=512,
            return_dense=True
        )["dense_vecs"][0].tolist()


def get_embeddings():
    return BGEM3Embedding()