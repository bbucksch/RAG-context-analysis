import os
import sys
import torch
import logging
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
from itertools import product
import random

def setup_environment():
    """Load environment variables and configure logging."""
    load_dotenv()
    HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')

    # If you get import errors for the dexter package, uncomment the next line
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    return HUGGING_FACE_TOKEN


def setup_device():
    """Set up the device for PyTorch (GPU or CPU)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    return device

def load_model_and_tokenizer(device, model_name):
    """Load the generative model and tokenizer."""
    logging.info("Loading generative model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # Ensure tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model

def load_dataset(config_path, reduce_corpus):
    """Load the retriever dataset."""
    logging.info("Loading dataset...")
    loader = RetrieverDataset("wikimultihopqa","wiki-musiqueqa-corpus", config_path, Split.DEV)
    queries, qrels, corpus, true_answers = loader.qrels()

    # queries = [Question1, Question2,...]
    # qrels = {query_id1: {"corpus_idx1": 1, "corpus_idx2": 1},...}
    # corpus = [Evidence1, Evidence2,...]
    # true_answers = [Answer1, Answer2,...]

    # Use a subset for testing purposes
    if reduce_corpus:
        corpus = corpus[:1000]
    queries = queries[:1200]

    return queries, qrels, corpus, true_answers

def generate_answer(query_text, retrieved_docs, tokenizer, model, device, randomize_order):
    """Generate an answer using the generative model."""
    if isinstance(retrieved_docs, list):
        if randomize_order:
            random.shuffle(retrieved_docs)

        retrieved_docs = " ".join(retrieved_docs)
    if not retrieved_docs:
        return ""


    # Few-shot prompt with chain-of-thought reasoning and strict retrieval focus
    combined_prompt = (
        "You are a helpful assistant that answers questions strictly based on the provided information. "
        "Your answers should be brief and directly address the question, such as a single word (e.g. 'Paris'), multiple words (e.g. 'King Egbert') or a simple 'yes' or 'no'.\n"
        "\n"
        "Q: Do both My Friend From The Park and Punks (Film) films have the directors from the same country?\n"
        "Information: My Friend from the Park is a 2015 Argentine drama film directed by Ana Katz. Punks is a 2000 film produced by Babyface, directed by Patrik-Ian Polk, and starring Rockmond Dunbar, Seth Gilliam, Renoly Santiago, Jazzmun, and Dwight Ewell.\n"
        "A: no\n"
        "\n"
        "Q: Which film was released first, Mawali No.1 or Vous Êtes De La Police?\n"
        "Information: Mawali No.1 is a 2002 Hindi- language Indian feature film directed by Leela V Prasad. Vous Êtes De La Police is a 2003 film.\n"
        "A: Mawali No.1\n"
        "\n"
        f"Q: {query_text}\n"
        f"Information: {retrieved_docs}\n"
        "Think step-by-step and provide 3 answers to the same question. Give one final answer based on the most voted answer. Give a brief answer strictly based on the provided information.\n"
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
    return answer.replace("A: ", "").strip(".")

def process_corpus_in_batches(corpus, batch_size):
    """Yield batches from the corpus."""
    for i in range(0, len(corpus), batch_size):
        yield corpus[i:i + batch_size]

def evaluate_retrieval(queries, qrels, corpus, true_answers, tokenizer, model, device, noise_ratios, randomize_order):
    """Evaluate the retrieval process and calculate exact match scores."""
    contrvr_search = Contriever(DenseHyperParams(
        query_encoder_path="facebook/contriever-msmarco",
        document_encoder_path="facebook/contriever-msmarco",
        batch_size=32,
        show_progress_bar=True
    ))
    similarity_measure = CosineSimilarity()
    metrics = RetrievalMetrics(k_values=[1, 3, 5])
    exact_match = CoverExactMatch()

    max_k = metrics.top_k
    max_nr = max(noise_ratios)
    docs_to_retrieve = max_k + max_nr * max_k + 10
    # Initial documents + noise documents + 10 possible noise documents that could be discarded

    all_responses = contrvr_search.retrieve(corpus, queries, docs_to_retrieve, similarity_measure, chunk=True, chunksize=500)
    # Dictionary of the form {query_id: {'corpus_idx': cos_score, 'corpus_idx': cos_score,...}, query_id: ...}
    sorted_responses = {
        query_id: dict(sorted(docs.items(), key=lambda x: x[1], reverse=True)) for query_id, docs in all_responses.items()
    }
    # Sort documents by cos score
    all_em_scores = {k: [] for k in metrics.k_values}

    for k, nr in product(metrics.k_values, noise_ratios):
        filtered_responses = {}
        # filtered_responses = {query_id1: {corpus_idx1: 1.0, corpus_idx2: 1.0}}

        for query_id, docs in sorted_responses.items():
            filtered_responses[query_id] = {}
            for i, doc_id in enumerate(docs):
                if i <= k:
                    filtered_responses[query_id][doc_id] = 1.0
                    # Keep first k documents
                elif i > k and len(filtered_responses[query_id]) < k*(nr+1):
                    # If number of documents for query hasn't reached k + nr*k
                    if doc_id not in list(qrels[query_id]):
                        # If document not in qrels (gold contexts)
                        filtered_responses[query_id][doc_id] = 1.0
                        # Add document to list
                else:
                    # If number of documents has reached k + nr*k
                    break

        em_scores = []
        for idx, (query_id, retrieved_docs) in enumerate(filtered_responses.items()):
            query_text = queries[idx].text() if queries[idx].id() == query_id else None
            logging.info(f"Query: {query_text}")

            retrieved_texts = [corpus[int(doc_id)].text() for doc_id in retrieved_docs]
            logging.info(f"POSSIBLE contexts: {[corpus[int(d)].text() for d in sorted_responses[query_id]]}")
            logging.info(f"k = {k} top contexts: {retrieved_texts[:k]}")
            logging.info(f"Negative nr*k = {nr*k} contexts: {retrieved_texts[k:]}")
            logging.info(f"Gold contexts: {[corpus[int(d)].text() for d in qrels[query_id] if int(d) < len(corpus)]}")

            true_answer = true_answers[idx].text()
            logging.info(f"True answer: {true_answer}")

            generated_answer = generate_answer(query_text, retrieved_texts, tokenizer, model, device, randomize_order)
            logging.info(f"Generated answer: {generated_answer}")

            em_score = exact_match.evaluate(generated_answer, true_answer)
            logging.info(f"Score: {em_score}")
            em_scores.append(em_score)

        avg_em_score = sum(em_scores) / len(em_scores) if em_scores else 0
        logging.info(f"Score for top-{k} contexts, noise ratio = {nr}: {avg_em_score}")
        all_em_scores[k].append((nr, avg_em_score))

    return all_em_scores

def main():
    """Main function to execute the retrieval and evaluation process."""
    # Step 1: Environment setup
    HUGGING_FACE_TOKEN = setup_environment()

    # Step 2: Device setup
    device = setup_device()

    # Step 3: Log in to Hugging Face
    logging.info("Logging into Hugging Face...")
    login(HUGGING_FACE_TOKEN)

    # Step 4: Load model and tokenizer
    model_name = "google/flan-t5-base"
    tokenizer, model = load_model_and_tokenizer(device, model_name)

    # Step 5: Load dataset
    config_path = "evaluation/config.ini"
    reduce_corpus = False # Do NOT reduce corpus!
    queries, qrels, corpus, true_answers = load_dataset(config_path, reduce_corpus)

    # Step 6: Evaluate retrieval with noise ratios
    noise_ratios = [1, 2] # Ratio noise:relevant contexts
    randomize_order = True # Randomize order of contexts in prompt?
    logging.info("Starting retrieval and evaluation...")
    all_em_scores = evaluate_retrieval(queries, qrels, corpus, true_answers,
                                       tokenizer, model, device, noise_ratios, randomize_order)

    # Step 7: Log the results
    logging.info("Summary of scores:")
    for k in all_em_scores:
        for nr, score in all_em_scores[k]:
            logging.info(f"Top-{k} contexts, noise ratio = {nr}: Score = {score}")
if __name__ == "__main__":
    main()
