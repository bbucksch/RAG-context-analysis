import os
import sys
import torch
import logging
import random
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from dexter.config.constants import Split
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.retriever.dense.Contriever import Contriever
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from dexter.utils.metrics.CoverExactMatch import CoverExactMatch

def setup_environment():
    load_dotenv()
    HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    return HUGGING_FACE_TOKEN

def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    return device

def load_model_and_tokenizer(device):
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model

def load_dataset(config_path):
    loader = RetrieverDataset(
        "wikimultihopqa",
        "wiki-musiqueqa-corpus",
        config_path,
        Split.DEV
    )
    queries, qrels, corpus, true_answers = loader.qrels()

    queries = queries[:1200]  # Limiting size for quicker runs

    return queries, qrels, corpus, true_answers

def generate_answer(query_text, retrieved_docs, tokenizer, model, device, randomize_order):
    if isinstance(retrieved_docs, list):
        if randomize_order:
            random.shuffle(retrieved_docs)
        retrieved_docs = " ".join(retrieved_docs)

    if not retrieved_docs:
        return ""

    combined_prompt = (
        "You are a helpful assistant that answers questions strictly based on the provided information. "
        "Your answers should be brief and directly address the question, such as a single word (e.g. 'Paris'), multiple words (e.g. 'King Egbert') or a simple 'yes' or 'no'.\n"
        "\n"
        "Q: Do both My Friend From The Park and Punks (Film) films have the directors from the same country?\n"
        "Information: My Friend from the Park is a 2015 Argentine drama film directed by Ana Katz. "
        "Punks is a 2000 film produced by Babyface, directed by Patrik-Ian Polk, and starring Rockmond Dunbar, Seth Gilliam, "
        "Renoly Santiago, Jazzmun, and Dwight Ewell.\n"
        "A: no\n"
        "\n"
        "Q: Which film was released first, Mawali No.1 or Vous Êtes De La Police?\n"
        "Information: Mawali No.1 is a 2002 Hindi-language Indian feature film directed by Leela V Prasad. "
        "Vous Êtes De La Police is a 2003 film.\n"
        "A: Mawali No.1\n"
        "\n"
        f"Q: {query_text}\n"
        f"Information: {retrieved_docs}\n"
        "Think step-by-step and provide 3 answers to the same question. Give one final answer based on the most voted answer. "
        "Give a brief answer strictly based on the provided information.\n"
    )

    inputs = tokenizer(
        combined_prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.replace("A: ", "").strip().strip(".")

def evaluate_noise_injection(queries, corpus, true_answers,
                             tokenizer, model, device,
                             randomize_order=True,
                             k_values=[1,3,5],
                             noise_ratios=[0,1,2,3]):
    contrvr_search = Contriever(DenseHyperParams(
        query_encoder_path="facebook/contriever-msmarco",
        document_encoder_path="facebook/contriever-msmarco",
        batch_size=32,
        show_progress_bar=True
    ))

    similarity_measure = CosineSimilarity()
    exact_match = CoverExactMatch()
    max_k = max(k_values)

    all_responses = contrvr_search.retrieve(
        corpus,
        queries,
        max_k,
        similarity_measure,
        chunk=True,
        chunksize=500
    )

    results = {k: {} for k in k_values}

    for k in k_values:
        topk_docs = {}
        for query_id, docs_scores in all_responses.items():
            sorted_docs = sorted(docs_scores.items(), key=lambda x: x[1], reverse=True)
            topk_doc_ids = [int(doc_id) for doc_id, _ in sorted_docs[:k]]
            topk_docs[query_id] = topk_doc_ids

        for noise_ratio in noise_ratios:
            em_scores = []

            for idx, query in enumerate(queries):
                query_id = query.id()
                query_text = query.text()
                true_answer = true_answers[idx].text()

                relevant_ids = topk_docs[query_id]

                full_doc_ids = list(range(len(corpus)))
                not_relevant_ids = [doc_i for doc_i in full_doc_ids if doc_i not in relevant_ids]

                if noise_ratio > 0 and len(not_relevant_ids) > 0:
                    random_doc_ids = random.sample(not_relevant_ids, min(noise_ratio, len(not_relevant_ids)))
                else:
                    random_doc_ids = []

                all_doc_ids = relevant_ids + random_doc_ids
                retrieved_texts = [corpus[doc_id].text() for doc_id in all_doc_ids]

                generated_answer = generate_answer(
                    query_text,
                    retrieved_texts,
                    tokenizer,
                    model,
                    device,
                    randomize_order
                )

                em_score = exact_match.evaluate(generated_answer, true_answer)
                em_scores.append(em_score)

            avg_em_score = sum(em_scores) / len(em_scores) if em_scores else 0
            results[k][noise_ratio] = avg_em_score

    return results

def main():
    HUGGING_FACE_TOKEN = setup_environment()
    device = setup_device()

    logging.info("Logging into Hugging Face...")
    if HUGGING_FACE_TOKEN:
        login(HUGGING_FACE_TOKEN)
    else:
        logging.warning("No Hugging Face token found in environment.")

    tokenizer, model = load_model_and_tokenizer(device)

    config_path = "evaluation/config.ini"
    queries, qrels, corpus, true_answers = load_dataset(config_path)

    logging.info(f"Number of queries: {len(queries)}")
    logging.info(f"Corpus size: {len(corpus)}")

    noise_ratios = [0, 1, 2, 3]
    k_values = [1, 3, 5]
    randomize_order = True

    results = evaluate_noise_injection(
        queries=queries,
        corpus=corpus,
        true_answers=true_answers,
        tokenizer=tokenizer,
        model=model,
        device=device,
        randomize_order=randomize_order,
        k_values=k_values,
        noise_ratios=noise_ratios
    )

    logging.info("\n===== RESULTS (CoverExactMatch) =====")
    for k in k_values:
        for nr in noise_ratios:
            score = results[k][nr]
            logging.info(f"Top-{k} relevant + {nr} noise -> CoverExactMatch = {score:.4f}")

    logging.info("Experiment 3 completed.")

if __name__ == "__main__":
    main()
