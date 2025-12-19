from retrieval_metrics import precision_at_k,recall_at_k
from faithfullness import faithfulness_score
from query import query_rag
from eval_data import EVAL_QUERIES
import matplotlib.pyplot as plt
import numpy as np


# metricc = ['Precision','Recall','Faithfulness']
# scores =

k=10

precision_list=[]
recall_list=[]
faithfullness_list=[]

for item in EVAL_QUERIES:
    query = item["query"]
    relevant_doc_ids = item["relevant_doc_ids"]
    
    llm_response, retrieved_docs = query_rag(query, k=k, return_docs=True)


    precision_list.append(precision_at_k(retrieved_docs, relevant_doc_ids, k))
    recall_list.append(recall_at_k(retrieved_docs, relevant_doc_ids, k))
    faithfullness_list.append(faithfulness_score(llm_response, retrieved_docs))


avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_faithfulness = np.mean(faithfullness_list)

# plotting the bar chart

plt.figure()
plt.bar(
    ["precision@K","recall@K","faithfulness"],
    [avg_precision, avg_recall, avg_faithfulness],
)

plt.ylim(0, 1)
plt.title("Overall RAG Evaluation Metrics")
plt.ylabel("Score")
plt.show()