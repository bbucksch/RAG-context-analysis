# Experiment 3: Randomly sample documents from the collection that are not relevant
# to the current query, then combine them with the top-k relevant documents as input
# to the LLM. Examine how noise injection affects final performance.

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
    """
    Load environment variables and configure logging.
    Expects a .env file containing HUGGING_FACE_TOKEN=<token>.
    """
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
    """
    Load the generative model (google/flan-t5-base) and tokenizer.
    We also add a padding token if the tokenizer lacks one.
    """
    logging.info("Loading generative model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # Ensure tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model

def load_dataset(config_path, reduce_corpus):
    """
    Load the RetrieverDataset, returning queries, qrels, corpus, and true_answers.
    We use the entire corpus as requested in the instructions.
    """
    logging.info("Loading dataset...")
    loader = RetrieverDataset("wikimultihopqa",
                              "wiki-musiqueqa-corpus",
                              config_path,
                              Split.DEV)
    queries, qrels, corpus, true_answers = loader.qrels()

    if reduce_corpus:
        corpus = corpus[:1000]

    return queries, qrels, corpus, true_answers

def generate_answer(query_text, retrieved_docs, tokenizer, model, device, randomize_order):
    """
    Generate an answer using the generative model with a few-shot prompt.
    The prompt is the same one used in experiment1.py.

    :param query_text: String of the question
    :param retrieved_docs: List of strings containing the text of retrieved documents
    :param tokenizer: The tokenizer
    :param model: The T5-based generative model
    :param device: 'cuda' or 'cpu'
    :param randomize_order: Whether to randomly shuffle the retrieved_docs
    :return: A string containing the model's answer
    """
    if isinstance(retrieved_docs, list):
        if randomize_order:
            random.shuffle(retrieved_docs)

        retrieved_docs = " ".join(retrieved_docs)

    if not retrieved_docs:
        return ""

    # The same few-shot prompt from experiment1
    combined_prompt = (
        "You are a helpful assistant that answers questions strictly based on the provided information. "
        "Your answers should be brief and directly address the question, such as a single word (e.g. 'Paris'), multiple words (e.g. 'King Egbert') or a simple 'yes' or 'no'.\n"
        "\n"
        "Q: Do both My Friend From The Park and Punks (Film) films have the directors from the same country?\n"
        "Information: My Friend from the Park is a 2015 Argentine drama film directed by Ana Katz. Punks is a 2000 film produced by Babyface, directed by Patrik-Ian Polk, and starring Rockmond Dunbar, Seth Gilliam, Renoly Santiago, Jazzmun, and Dwight Ewell.\n"
        "A: no\n"
        "\n"
        "Q: Which film was released first, Mawali No.1 or Vous Êtes De La Police?\n"
        "Information: Mawali No.1 is a 2002 Hindi-language Indian feature film directed by Leela V Prasad. Vous Êtes De La Police is a 2003 film.\n"
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
    return answer.replace("A: ", "").strip().strip(".")


def random_docs(len_corpus, relevant_docs, k, nr):
    random_doc_ids = []
    for _ in range(nr * k):
        number = random.randint(0, len_corpus - 1)
        while number in relevant_docs:
            number = random.randint(0, len_corpus - 1)
        random_doc_ids.append(number)

    return random_doc_ids


def evaluate_noise_injection(queries, corpus, true_answers,
                             tokenizer, model, device,
                             randomize_order,
                             noise_ratios,
                             k_values=[1,3,5]):
    """
    Retrieve top-k relevant documents for each query using Contriever,
    then inject random negative (unrelated) documents in various ratios.
    Evaluate performance (CoverExactMatch) for each (k, noise_ratio) combination.

    :param queries: List of Query objects
    :param corpus: List of Evidence objects
    :param true_answers: List of ground truth answer strings
    :param tokenizer: Hugging Face tokenizer
    :param model: Hugging Face T5 model
    :param device: PyTorch device
    :param randomize_order: Whether to shuffle contexts in the final prompt
    :param k_values: The top-k values to test
    :param noise_ratios: The number of random noise docs to add
    :return: Nested dictionary of {k: {noise_ratio: average_EM_score}}
    """
    # Initialize the retriever
    contrvr_search = Contriever(DenseHyperParams(
        query_encoder_path="facebook/contriever-msmarco",
        document_encoder_path="facebook/contriever-msmarco",
        batch_size=32,
        show_progress_bar=True
    ))
    similarity_measure = CosineSimilarity()
    metrics = RetrievalMetrics(k_values=k_values)

    # Use cover exact match
    exact_match = CoverExactMatch()

    # We need to retrieve up to max_k contexts for each query
    max_k = metrics.top_k

    # Step 1: Perform retrieval once for all queries
    all_responses = contrvr_search.retrieve(
        corpus,
        queries,
        max_k,
        similarity_measure,
        chunk=True,
        chunksize=500
    )
    # all_responses is a dict: {query_id: {doc_id: cos_sim, ...}, ...}

    # Prepare a structure to store final results
    # results[k][noise_ratio] = average EM
    results = {k: {} for k in metrics.k_values}

    # For each possible k in k_values, we filter the top-k relevant docs
    for k in metrics.k_values:
        # Build a dict of doc IDs for each query
        topk_docs = {}
        for query_id, docs_scores in all_responses.items():
            # Sort documents by their similarity (descending)
            # then keep the top k doc IDs
            sorted_docs = sorted(docs_scores.items(), key=lambda x: x[1], reverse=True)
            topk_doc_ids = [int(doc_id) for doc_id, _ in sorted_docs[:k]]
            topk_docs[query_id] = topk_doc_ids

        # Evaluate for different noise ratios
        for noise_ratio in noise_ratios:
            em_scores = []

            for idx, query in enumerate(queries):
                query_id = query.id()
                query_text = query.text()
                true_answer = true_answers[idx].text()

                # 1) Get the top-k doc IDs for this query
                relevant_ids = topk_docs[query_id]

                # 2) Sample random (noise_ratio) docs from the corpus that are *not* in relevant_ids
                random_doc_ids = random_docs(len(corpus), relevant_ids, k, noise_ratio)

                # 3) Combine the text of relevant docs + random docs
                all_doc_ids = relevant_ids + random_doc_ids
                retrieved_texts = [corpus[doc_id].text() for doc_id in all_doc_ids]

                # 4) Pass to LLM
                generated_answer = generate_answer(query_text,
                                                  retrieved_texts,
                                                  tokenizer,
                                                  model,
                                                  device,
                                                  randomize_order)

                # 5) Evaluate CoverExactMatch
                em_score = exact_match.evaluate(generated_answer, true_answer)
                em_scores.append(em_score)

                if idx%100 == 0:
                    logging.info(f"k = {k}, nr = {noise_ratio}, idx = {idx}, em_score = {em_score}\n"
                                 f"Query = {query_text}\nGenerated Answer = {generated_answer}\nTrue Answer = {true_answer}")
                if idx%600 == 0:
                    logging.info(f"Top-{k} relevant docs: {[corpus[doc].text() for doc in topk_docs[query_id]]}")
                    logging.info(f"Top-{k} relevant docs (same?): {[corpus[int(doc)].text() for doc, _ in sorted(all_responses[query_id].items(), key=lambda x: x[1], reverse=True)[:k]]}")
                    logging.info(f"{k*noise_ratio} noise docs: {[corpus[doc].text() for doc in random_doc_ids]}")
                    logging.info(f"Total contexts: {retrieved_texts}")

            # Compute average EM over all queries
            avg_em_score = sum(em_scores) / len(em_scores) if em_scores else 0
            results[k][noise_ratio] = avg_em_score

    return results

def main():
    # Step 1: Environment setup
    HUGGING_FACE_TOKEN = setup_environment()

    # Step 2: Device setup
    device = setup_device()

    # Step 3: Log in to Hugging Face
    logging.info("Logging into Hugging Face...")
    if HUGGING_FACE_TOKEN:
        login(HUGGING_FACE_TOKEN)
    else:
        logging.warning("No Hugging Face token found in environment.")

    # Step 4: Load model and tokenizer
    model_name = "google/flan-t5-base"
    tokenizer, model = load_model_and_tokenizer(device, model_name)

    # Step 5: Load dataset
    config_path = "evaluation/config.ini"
    reduce_corpus = False
    queries, qrels, corpus, true_answers = load_dataset(config_path, reduce_corpus)

    logging.info(f"Number of queries: {len(queries)}")
    logging.info(f"Corpus size: {len(corpus)}")

    # Step 6: Evaluate retrieval with noise injection
    logging.info("Starting retrieval + noise injection evaluation...")
    # Decide how many random docs to inject
    noise_ratios = [1, 2, 3]   # Feel free to adjust
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
        noise_ratios=noise_ratios,
        k_values=k_values
    )

    # Step 7: Print the results
    logging.info("\n===== RESULTS (CoverExactMatch) =====")
    for k in k_values:
        for nr in noise_ratios:
            score = results[k][nr]
            logging.info(f"Top-{k} relevant docs + {nr} noise ratio -> CoverExactMatch = {score:.4f}")

    logging.info("Experiment 3 completed.")

if __name__ == "__main__":
    main()
