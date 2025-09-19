import os
import sys
import torch
import logging
import random
import time
from dotenv import load_dotenv
from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.utils.metrics.ExactMatch import ExactMatch
from huggingface_hub import login
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm  # Progress bar

# Load environment variables (Hugging Face token)
load_dotenv()
HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_answer(query_text, retrieved_docs, tokenizer, model, device):
    if isinstance(retrieved_docs, list):
        retrieved_docs = " ".join(retrieved_docs)
    if not retrieved_docs:
        return ""
    
    # Few-shot prompt with chain-of-thought reasoning and strict retrieval focus
    combined_prompt = (
        "You are a helpful assistant that answers questions strictly based on the provided information. "
        "Your answers should be brief and directly address the question, such as a single word (e.g. 'Paris'), "
        "multiple words (e.g. 'King Egbert') or a simple 'yes' or 'no'.\n"
        "\n"
        "Q: Do both My Friend From The Park and Punks (Film) films have the directors from the same country?\n"
        "Information: My Friend from the Park is a 2015 Argentine drama film directed by Ana Katz. "
        "Punks is a 2000 film produced by Babyface, directed by Patrik-Ian Polk, and starring Rockmond Dunbar, "
        "Seth Gilliam, Renoly Santiago, Jazzmun, and Dwight Ewell.\n"
        "A: no\n"
        "\n"
        "Q: Which film was released first, Mawali No.1 or Vous Êtes De La Police?\n"
        "Information: Mawali No.1 is a 2002 Hindi-language Indian feature film directed by Leela V Prasad. "
        "Vous Êtes De La Police is a 2003 film.\n"
        "A: Mawali No.1\n"
        "\n"
        f"Q: {query_text}\n"
        f"Information: {retrieved_docs}\n"
        "Think step-by-step and provide 3 answers to the same question. Give one final answer based on the most "
        "voted answer. Give a brief answer strictly based on the provided information.\n"
    )
    
    # Tokenize and generate the answer
    inputs = tokenizer(
        combined_prompt,
        return_tensors="pt",
        truncation=True,  # Explicitly enable truncation
        padding="longest"
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=100,
        min_length=1,
        num_beams=3,
        early_stopping=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.debug(f"Generated answer: {answer}")
    return answer

if __name__ == "__main__":
    logging.info("Logging into Hugging Face...")
    login(HUGGING_FACE_TOKEN)

    config_path = "evaluation/config.ini"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Dataset loading
    try:
        start_time = time.time()
        logging.info("Loading dataset...")
        print("Loading dataset... (this can take a while)")
        loader = RetrieverDataset("wikimultihopqa", "wiki-musiqueqa-corpus", config_path, Split.DEV)
        print(f"Dataset loaded successfully in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        sys.exit(1)

    # Extract queries and corpus
    try:
        start_time = time.time()
        print("Starting query and corpus extraction...")
        queries, qrels, corpus = loader.qrels()
        print(f"Queries extracted: {len(queries)}")
        queries = queries[:1]  # Reduced queries for faster debugging
        print(f"Using a reduced set of queries: {len(queries)}")

        print(f"Number of corpus documents: {len(corpus)}")
        corpus = corpus[:1]  # Reduced corpus size for debugging
        print(f"Using a reduced corpus size: {len(corpus)}")

        print("Transforming corpus texts...")
        corpus_texts = [c.text() for c in corpus]
        print(f"Corpus texts transformed in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        logging.error(f"Error extracting queries/corpus: {e}")
        sys.exit(1)

    # Retrieve all true answers
    try:
        true_answers = loader.base_dataset.answers()
        print("True answers loaded.")
    except Exception as e:
        logging.error(f"Error loading answers: {e}")
        sys.exit(1)

    # Load generative model
    try:
        start_time = time.time()
        logging.info("Loading generative model...")
        print("Initializing model...")
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        model.gradient_checkpointing_enable()

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
        print(f"Generative model loaded successfully in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        logging.error(f"Error loading generative model: {e}")
        sys.exit(1)

    # Initialize ExactMatch metric
    exact_match = ExactMatch(match_type="cover")
    k_values = [1, 3, 5]
    noise_ratios = [0, 1, 2]

    # Initialize results dictionary to store EM scores for each (k, noise_ratio) combination
    results = {(k, nr): [] for k in k_values for nr in noise_ratios}

    # Process queries
    print("Processing queries... (with reduced data)")
    for idx, q in enumerate(tqdm(queries, desc="Queries Processed")):
        try:
            query_id = q.id()
            query_text = q.text()
            true_answer = true_answers.get(query_id, None)

            if not true_answer:
                logging.warning(f"No true answer for query_id {query_id}")
                continue

            # Get oracle contexts
            oracle_contexts = loader.base_dataset.get_context_by_id(query_id)
            oracle_texts = [x for entry in oracle_contexts for x in entry[1]]

            # Create a set for oracle texts to ensure no duplicates
            oracle_texts_set = set(oracle_texts)

            # Candidate noise docs
            noise_candidates = [doc for doc in corpus_texts if doc not in oracle_texts_set]

            for k in k_values:
                relevant_docs = oracle_texts[:k]

                for nr in noise_ratios:
                    noise_count = nr * k
                    noise_docs = random.sample(noise_candidates, noise_count) if noise_count > 0 else []

                    final_docs = relevant_docs + noise_docs

                    print(
                        f"Processing Query ID {query_id} - top-{k}, noise_ratio={nr}, "
                        f"noise_docs={noise_count}."
                    )

                    generated_answer = generate_answer(query_text, final_docs, tokenizer, model, device)
                    em_score = exact_match.evaluate(generated_answer, true_answer)
                    results[(k, nr)].append(em_score)
        except Exception as e:
            logging.error(f"Error processing query {idx + 1}/{len(queries)}: {e}")

    # Compute average EM scores
    avg_em_scores = {
        (k, nr): (sum(scores)/len(scores) if scores else 0)
        for (k, nr), scores in results.items()
    }

    logging.info("Summary of Exact Match scores:")
    for k in k_values:
        for nr in noise_ratios:
            score = avg_em_scores[(k, nr)]
            logging.info(f"Top-{k} contexts + noise_ratio={nr}: Exact Match score = {score}")

    print("Experiment complete!")
