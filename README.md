# 10-701-Project-Implementation: Assessing Factuality of Generated Answers using RAG-based Question Answering Pipeline

## RAG-based pipeline for Question Answering

This repository contains the implementation of a pipeline for Question Answering using the RAG model. The pipeline is based on the following steps:
- **Indexing**: Index a corpus of documents using an embedding model and vector store.
- **Retrieval**: Retrieve relevant documents from a corpus using a retriever model.
- **Reading**: Read the retrieved documents and extract the answer to the question using a reader model.
- **Generation**: Generate an answer to the question using the retrieved documents and the question.
- **Evaluation**: Evaluate the generated answer by comparing it to the ground truth answer and assessing its quality using an evaluator model.

The pipeline is implemented in Python using the Hugging Face Transformers library and the Faiss library for similarity search.
- **Hugging Face Transformers**: https://huggingface.co/transformers/
- **Faiss**: https://python.langchain.com/docs/integrations/vectorstores/faiss/
- **LLMs**: GPT-4, Zephyr-7B, Mistral 8x7B

## Experiments

We evaluate the pipeline on the [Hugging Face documentation question answering dataset](https://huggingface.co/datasets/m-ric/huggingface_doc_qa_eval) using the following metrics:

- **Accuracy**: Percentage of questions for which the generated answer matches the ground truth answer (corresponding to an evaluation score above 4/5).
- **F1 Score**: F1 score of the generated answer compared to the ground truth answer.
- **BLEU Score**: BLEU score of the generated answer compared to the ground truth answer.

We test configurations of the pipeline by modifying the following parameters:

- **Number of Retrieved Documents**: Number of documents retrieved by the retriever model -> [5, 10, 20, 40].
- **Chunk Size**: Size of the document chunks stored in the index in tokens -> [128, 256, 512, 1024].

