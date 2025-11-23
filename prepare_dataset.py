import re
import json
from datasets import load_dataset
from tqdm import tqdm

# -----------------------------------------------
#  Expert Heuristic Rules
# -----------------------------------------------

FACTUAL_CUES = [
    r"who\b", r"when\b", r"where\b", r"what\b", r"name\b",
    r"capital", r"born", r"located", r"population"
]

MULTIHOP_CUES = [
    r"between", r"relationship", r"related to",
    r"after.*before", r"before.*after",
    r"both", r"compare", r"connection"
]

NUMERICAL_CUES = [
    r"how many", r"how much", r"distance", r"cost",
    r"percentage", r"percent", r"years", r"age",
    r"duration", r"height", r"weight"
]

COMMONSENSE_CUES = [
    r"why", r"reason", r"cause", r"purpose", r"should",
    r"would", r"likely", r"typically"
]


# -----------------------------------------------
#  Expert Assignment Function
# -----------------------------------------------

def classify_expert(question):
    q = question.lower()

    # Numerical first because it's specific
    if any(re.search(pattern, q) for pattern in NUMERICAL_CUES):
        return "numerical_reasoning"

    # Multi-hop often has relational patterns
    if any(re.search(pattern, q) for pattern in MULTIHOP_CUES):
        return "multi_hop_reasoning"

    # Commonsense / pragmatic reasoning
    if any(re.search(pattern, q) for pattern in COMMONSENSE_CUES):
        return "commonsense_reasoning"

    # Default: factual lookup
    if any(re.search(pattern, q) for pattern in FACTUAL_CUES):
        return "factual_lookup"

    return "factual_lookup"  # safe fallback


# -----------------------------------------------
#  Download Dataset
# -----------------------------------------------

print("Downloading NQ dataset...")
nq = load_dataset("nq_open", split="train")

# -----------------------------------------------
#  Annotate Dataset with Expert Type
# -----------------------------------------------

output_path = "nq_annotated_moe.jsonl"
print(f"Annotating and saving to {output_path} ...")

with open(output_path, "w") as f:
    for item in tqdm(nq):
        q = item["question"]
        expert = classify_expert(q)

        record = {
            "question": item["question"],
            "answer": item["answer"],
            "expert_label": expert
        }

        f.write(json.dumps(record) + "\n")

print("Done!")
print(f"Saved annotated dataset to {output_path}")
