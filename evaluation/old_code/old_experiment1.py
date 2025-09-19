import os
import sys

# I was getting import errors for the dexter package
# If you don't please comment this line, don't remove it
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import logging
from dotenv import load_dotenv
from dexter.config.constants import Split
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.retriever.dense.ANCE import ANCE
from dexter.retriever.dense.Contriever import Contriever
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from dexter.utils.metrics.ExactMatch import ExactMatch
from huggingface_hub import login
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import random

# Install using "pip install python-dotenv"
load_dotenv()
HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_answer(query_text, retrieved_docs):
    logging.debug(f"Generating answer for query: {query_text} and retrieved_docs: {retrieved_docs}")
    if isinstance(retrieved_docs, list):
        retrieved_docs = " ".join(retrieved_docs)
    if not retrieved_docs:
        return ""
    combined_input = f"Answer the following question briefly: {query_text} Using the following information when useful: {retrieved_docs}"
    inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model.generate(**inputs, max_length=50, min_length=1, num_beams=3, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # logging.debug(f"Generated answer: {answer}")
    return answer


def process_corpus_in_batches(corpus, batch_size):
    for i in range(0, len(corpus), batch_size):
        yield corpus[i:i + batch_size]


# Main
if __name__ == "__main__":
    # Log in to Hugging Face
    logging.info("Logging into Hugging Face...")
    login(HUGGING_FACE_TOKEN)

    # Path to the configuration file
    config_path = "evaluation/config.ini"

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load dataset
    logging.info("Loading dataset...")
    loader = RetrieverDataset("wikimultihopqa","wiki-musiqueqa-corpus","evaluation/config.ini",Split.DEV)

    # Retrieve queries, qrels, and corpus
    queries, qrels, corpus, true_answers = loader.qrels()

    # Qrels empty?

    # Retrieve queries
    query_dict = {q.id(): q.text() for q in queries}

    # Retrieve true answers from the dataset
    logging.info("Retrieving true answers from dataset...")
    # true_answers = loader.base_dataset.answers()  # Use the answers method from MyDataLoader

    # Select first 1200 queries for the test set
    # queries = queries[:1200]
    corpus = corpus[:1000]

    # Print some queries and corpus for debugging
    logging.debug(f"Sample Queries: {[query.text() for query in queries[:5]]}")
    logging.debug(f"Sample Corpus: {[evidence.text() for evidence in corpus[:5]]}")

    # Load generative model
    logging.info("Loading generative model...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)
    model.gradient_checkpointing_enable()

    # Ensure tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Set batch size for processing corpus
    corpus_batch_size = 500

    # # Retrieval and evaluation
    # contrvr_search = Contriever(DenseHyperParams(
    #     query_encoder_path="sentence-transformers/all-MiniLM-L6-v2",  # Vervang door een kleiner model
    #     document_encoder_path="sentence-transformers/all-MiniLM-L6-v2",
    #     batch_size=16,
    #     show_progress_bar=True
    # ))
    # Retrieval and evaluation with a larger model
    contrvr_search = Contriever(DenseHyperParams(
        query_encoder_path="sentence-transformers/all-distilroberta-v1",  # Replace with a larger model
        document_encoder_path="sentence-transformers/all-distilroberta-v1",
        batch_size=16,
        show_progress_bar=True
    ))
    similarity_measure = CosineSimilarity()
    k_values = [1, 3, 5]
    metrics = RetrievalMetrics(k_values=k_values)
    exact_match = ExactMatch()

    all_em_scores = []

    for k in k_values:
        logging.info(f"Retrieving top-{k} contexts...")
        all_responses = {}

        for corpus_batch in process_corpus_in_batches(corpus, corpus_batch_size):
            response = contrvr_search.retrieve(corpus_batch, queries, k, similarity_measure, chunk=True, chunksize=1000)
            all_responses.update(response)

        # Print some retrieved documents for debugging
        logging.debug(f"Sample Retrieved Documents for top-{k}: {list(all_responses.items())[:5]}")

        # Check if retrieval is successful
        if not all_responses:
            logging.error(f"No documents retrieved for top-{k} contexts.")
            continue

        # Retrieval metrics evaluation
        retrieval_metrics = metrics.evaluate_retrieval(qrels=qrels, results=all_responses)
        logging.info(f"Retrieval metrics for top-{k}: {retrieval_metrics}")

        # Exact Match evaluation
        em_scores = []
        for query_id, retrieved_docs in all_responses.items():
            logging.debug(f"Query ID: {query_id}")
            query_text = query_dict[query_id]
            # query_text = next((query.text() for query in queries if query.id() == query_id), None)
            logging.debug(f"Query Text: {query_text}")

            # Retrieve texts of documents
            retrieved_texts = [corpus[int(doc_id)].text() for doc_id in retrieved_docs if int(doc_id) < len(corpus)]
            # retrieved_texts = [corpus[int(doc_id)].text() for doc_id in retrieved_docs]

            logging.debug(f"Retrieved texts for query_id {query_id}: {retrieved_texts}")
            if not retrieved_texts:
                logging.error(f"No texts retrieved for query_id {query_id}")
                continue

            # Generate answer
            generated_answer = generate_answer(query_text, retrieved_texts)
            logging.debug(f"Generated Answer: {generated_answer}")

            # Ensure true_answers is correctly indexed by query IDs
            logging.debug(f"True answer: {true_answers[query_id]}")
            if not true_answers[query_id]:
                logging.error(f"No true answer available for query_id {query_id}")
                continue

            # Evaluate Exact Match
            em_score = exact_match.evaluate(generated_answer, true_answers[query_id])
            logging.debug(f"Exact Match score for query_id {query_id}: {em_score}")
            em_scores.append(em_score)

        avg_em_score = sum(em_scores) / len(em_scores) if em_scores else 0
        logging.info(f"Exact Match score for top-{k}: {avg_em_score}")
        all_em_scores.append((k, avg_em_score))

    # Summary of scores
    logging.info("Summary of Exact Match scores:")
    for k, score in all_em_scores:
        logging.info(f"Top-{k} contexts: Exact Match score = {score}")

