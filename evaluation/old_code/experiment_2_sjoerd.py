import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import logging
from dotenv import load_dotenv
from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.utils.metrics.ExactMatch import ExactMatch
from huggingface_hub import login
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_answer(query_text, retrieved_docs):
    if isinstance(retrieved_docs, list):
        retrieved_docs = " ".join(retrieved_docs)
    if not retrieved_docs:
        return ""
    combined_input = f"Answer the following question briefly: {query_text} Using the following information when useful: {retrieved_docs}"
    inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model.generate(**inputs, max_length=50, min_length=1, num_beams=3, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.debug(f"Generated answer: {answer}")
    return answer

if __name__ == "__main__":
    logging.info("Logging into Hugging Face...")
    login(HUGGING_FACE_TOKEN)

    config_path = "evaluation/config.ini"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    logging.info("Loading dataset...")
    loader = RetrieverDataset("my-dataset", "my-dataset-corpus", config_path, Split.DEV, tokenizer=None)
    queries, qrels, corpus = loader.qrels()

    # Retrieve the official answers
    true_answers = loader.base_dataset.answers()

    # Limit queries and corpus as needed
    queries = queries[:10]
    corpus = corpus[:1000]

    # Load model
    logging.info("Loading generative model...")
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.gradient_checkpointing_enable()

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    exact_match = ExactMatch()
    k_values = [1, 3, 5]
    all_em_scores = {k: [] for k in k_values}

    # No retrieval here, just oracle contexts
    for q in queries:
        query_id = q.id()
        query_text = q.text()
        true_answer = true_answers[query_id]

        if not true_answer:
            logging.warning(f"No true answer for query_id {query_id}")
            continue

        # Get oracle contexts directly
        oracle_contexts = loader.base_dataset.get_context_by_id(query_id)
        retrieved_texts = [x for entry in oracle_contexts for x in entry[1]]

        for k in k_values:
            logging.info(f"Evaluating with top-{k} oracle contexts...")
            retrieved_batch = retrieved_texts[:k]
            generated_answer = generate_answer(query_text, retrieved_batch)
            em_score = exact_match.evaluate(generated_answer, true_answer)
            all_em_scores[k].append(em_score)

    # Compute averages
    avg_em_scores = {k: (sum(scores)/len(scores) if scores else 0) for k, scores in all_em_scores.items()}

    logging.info("Summary of Exact Match scores (oracle-only):")
    for k, score in avg_em_scores.items():
        logging.info(f"Top-{k} contexts: Exact Match score = {score}")
