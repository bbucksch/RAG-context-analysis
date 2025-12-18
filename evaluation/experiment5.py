import os
import sys
import torch
import logging
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

from dexter.config.constants import Split
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.retriever.dense.Contriever import Contriever
from dexter.retriever.dense.ADORE import ADORERetriever
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from dexter.utils.metrics.CoverExactMatch import CoverExactMatch


def setup_environment():
    """Load environment variables and configure logging."""
    load_dotenv()
    HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return HUGGING_FACE_TOKEN


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device


def load_model_and_tokenizer(device):
    """Load T5 model and tokenizer from Hugging Face."""
    logging.info("Loading Flan-T5 model...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)

    # Handle missing pad token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


def load_dataset(config_path):
    """
    Load queries, qrels, corpus, and true_answers from the RetrieverDataset.
    By default, we reduce queries to 2 items to avoid dimension errors with Contriever.
    """
    loader = RetrieverDataset("wikimultihopqa", "wiki-musiqueqa-corpus", config_path, Split.DEV, tokenizer=None)
    queries, qrels, corpus, true_answers = loader.qrels()

    corpus = corpus[:1000]
    # Limit to two queries to avoid single-query dimension issues in Contriever
    # queries = queries[:2]
    # corpus = corpus[:2]
    queries = queries[:1200]

    return queries, qrels, corpus, true_answers


def generate_answer(query_text, retrieved_docs, tokenizer, model, device):
    """Generate an answer using the generative model."""
    if isinstance(retrieved_docs, list):
        retrieved_docs = " ".join(retrieved_docs)
    if not retrieved_docs:
        return ""

    prompt = (
        "You are a helpful assistant that answers questions strictly based on the provided information. "
        "Your answers should be brief and directly address the question.\n\n"
        "Q: Do both My Friend From The Park and Punks (Film) films have directors from the same country?\n"
        "Information: My Friend from the Park is a 2015 Argentine drama film directed by Ana Katz. "
        "Punks is a 2000 film produced by Babyface, directed by Patrik-Ian Polk.\n"
        "A: no\n\n"
        "Q: Which film was released first, Mawali No.1 or Vous Êtes De La Police?\n"
        "Information: Mawali No.1 is a 2002 Hindi-language film. Vous Êtes De La Police is a 2003 film.\n"
        "A: Mawali No.1\n\n"
        f"Q: {query_text}\n"
        f"Information: {retrieved_docs}\n"
        "Think step-by-step and provide 3 answers to the same question. Give one final answer based on the most voted answer. "
        "Give a brief answer strictly based on the provided information.\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=50)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.replace("A: ", "").strip(".")


def evaluate_single_retriever(queries, corpus, true_answers, tokenizer, model, device, retriever):
    """Evaluate exact-match scores for a given retriever's responses."""
    exact_match = CoverExactMatch()
    em_scores_for_k = []
    k_values=[1, 3, 5]
    max_k = max(k_values)

    similarity = CosineSimilarity()

    responses = retriever.retrieve(corpus, queries, max_k, similarity)

    for k in k_values:
        em_scores = []
        filtered_responses = {
            query_id: {
                doc_id: 1.0 for doc_id in [d for d,cos in sorted(docs.items(), key=lambda x: x[1], reverse=True)][:k]
                # Sort documents by cos score
                # Non-sorted version:
                # doc_id: 1.0 for doc_id in list(docs)[:k]
                # filtered_responses = {query_id1: {corpus_idx1: 1.0, corpus_idx2: 1.0}}
            } for query_id, docs in responses.items()
        }

        for idx, (qid, doc_ids) in enumerate(filtered_responses.items()):
            query_text = next((q.text() for q in queries if q.id() == qid), "")
            retrieved_texts = [corpus[int(did)].text() for did in doc_ids if int(did) < len(corpus)]

            # Collect gold answers matching this query's id
            '''
            gold_answers = []
            for ans in true_answers:
                # 'ans.id' is the attribute that holds the matching query_id
                # 'ans.text' is the gold answer text
                if (ans.id if not callable(ans.id) else ans.id()) == qid:
                    text_val = ans.text if not callable(ans.text) else ans.text()
                    gold_answers.append(text_val)
            '''
            true_answer = true_answers[idx].text()
            generated_answer = generate_answer(query_text, retrieved_texts, tokenizer, model, device)
            score = exact_match.evaluate(generated_answer, true_answer)
            print("------------------------------------------------")
            print(f"Question: {query_text}")
            print(f"Predicted answer: {generated_answer}")
            print(f"True answer: {true_answer}")
            print("------------------------------------------------")
            em_scores.append(score)

        mean_em = sum(em_scores) / len(em_scores) if em_scores else 0
        em_scores_for_k.append((k, mean_em))

    return em_scores_for_k


def main():
    import argparse
    # Parse argument
    parser = argparse.ArgumentParser(description="Run retrieval experiments with ADORE or Contriever.")
    parser.add_argument(
        "Select retriever: 0 -> ADORE, 1 -> Contriever",
        type=int,
        choices=[0, 1],
        help="Set to 0 to run with ADORE, or 1 to run with Contriever."
    )
    args = parser.parse_args()

    # Setup environment and device
    HUGGING_FACE_TOKEN = setup_environment()
    device = setup_device()

    logging.info("Logging into Hugging Face...")
    login(HUGGING_FACE_TOKEN)

    tokenizer, model = load_model_and_tokenizer(device)
    config_path = "evaluation/config.ini"

    logging.info("Loading dataset...")
    queries, qrels, corpus, true_answers = load_dataset(config_path)

    # Choose the retriever based on the argument
    if args.retriever == 0:
        logging.info("Running retrieval experiment with ADORE...")
        adore = ADORERetriever("zycao/adore-star", device, batch_size=64)

        logging.info("Retrieving with ADORE...")
        adore_scores = evaluate_single_retriever(queries, corpus, true_answers, tokenizer, model, device, adore)

        logging.info("Exact Match Scores for ADORE:")
        for k, score in adore_scores:
            logging.info(f"Top-{k} => Exact Match: {score:.3f}")

    else:
        logging.info("Running retrieval experiment with Contriever...")
        contriever = Contriever(DenseHyperParams(
            query_encoder_path="facebook/contriever-msmarco",
            document_encoder_path="facebook/contriever-msmarco",
            batch_size=32,
            show_progress_bar=False
        ))

        logging.info("Retrieving with Contriever...")
        contriever_scores = evaluate_single_retriever(queries, corpus, true_answers, tokenizer, model, device, contriever)

        logging.info("Exact Match Scores for Contriever:")
        for k, score in contriever_scores:
            logging.info(f"Top-{k} => Exact Match: {score:.3f}")


if __name__ == "__main__":
    main()
