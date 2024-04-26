
from langchain_community.llms import HuggingFaceEndpoint
from multiprocessing import freeze_support
import os
import datasets
from benchmark import evaluate_answers, run_rag_tests, EVALUATION_PROMPT
from index_creation import load_embeddings
from langchain.docstore.document import Document as LangchainDocument
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage
from langchain_openai import ChatOpenAI
from tqdm import tqdm

if __name__ == '__main__':
    #freeze_support ()

    # Load knowledge base
    ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")

    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
        for doc in tqdm(ds)
    ]

    # Load RAG reader model
    repo_id = "HuggingFaceH4/zephyr-7b-beta"
    READER_MODEL_NAME = "zephyr-7b-beta"

    READER_LLM = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        max_new_tokens=512,
        top_k=30,
        temperature=0.1,
        repetition_penalty=1.03,
    )

    eval_dataset = datasets.load_dataset("m-ric/huggingface_doc_qa_eval", split="train")

    evaluation_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are a fair evaluator language model."),
            HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT),
        ]
    )

    eval_chat_model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
    evaluator_name = "GPT4"

    if not os.path.exists("./output"):
        os.mkdir("./output")

    chunk_size = 200
    #num_docs = 10 # Number of documents to retrieve, uncomment to test with a fixed number of documents and vary chunk size
    # for baseline llm set num_docs to 0

    #for chunk_size in [50, 100, 200, 400]:
    for num_docs in [5, 10, 20, 40]: 
        for embeddings in ["thenlper/gte-small"]:
            settings_name = f"chunk_size:{chunk_size}_num-docs:{num_docs}_embeddings:{embeddings.replace('/', '~')}_reader-model:{READER_MODEL_NAME}"
            output_file_name = f"./output/rag_{settings_name}.json"

            print(f"Running evaluation for {settings_name}:")

            print("Loading knowledge base embeddings...")
            knowledge_index = load_embeddings(
                RAW_KNOWLEDGE_BASE,
                chunk_size=chunk_size,
                embedding_model_name=embeddings,
            )

            print("Running RAG tests...")
            run_rag_tests(
                eval_dataset=eval_dataset,
                llm=READER_LLM,
                knowledge_index=knowledge_index,
                output_file=output_file_name,
                verbose=False,
                test_settings=settings_name,
                num_retrieved_docs=num_docs,
            )

            print("Running evaluation...")
            evaluate_answers(
                output_file_name,
                eval_chat_model,
                evaluator_name,
                evaluation_prompt_template,
            )