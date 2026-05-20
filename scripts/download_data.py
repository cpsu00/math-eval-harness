"""Download math evaluation datasets into Uni-DPO JSONL format."""
import argparse
import json
import os
from datasets import load_dataset, concatenate_datasets


def save_jsonl(path, examples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} examples to {path}")


def download_math500(data_dir):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    examples = [
        {"idx": i, "question": ex["problem"], "solution": ex["solution"], "type": ex["subject"]}
        for i, ex in enumerate(ds)
    ]
    save_jsonl(f"{data_dir}/math500/test.jsonl", examples)


def download_minerva_math(data_dir):
    subjects = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
    ]
    splits = [load_dataset("EleutherAI/hendrycks_math", s, split="test") for s in subjects]
    ds = concatenate_datasets(splits)
    examples = [
        {"idx": i, "question": ex["problem"], "solution": ex["solution"], "type": ex["type"]}
        for i, ex in enumerate(ds)
    ]
    save_jsonl(f"{data_dir}/minerva_math/test.jsonl", examples)


def download_aime24(data_dir):
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    examples = [
        {"idx": i, "question": ex["Problem"], "answer": str(ex["Answer"])}
        for i, ex in enumerate(ds)
    ]
    save_jsonl(f"{data_dir}/aime24/test.jsonl", examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="Math/evaluation/data")
    parser.add_argument("--datasets", default="math500,minerva_math,aime24",
                        help="Comma-separated list. gsm8k loads from HF automatically.")
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    for name in datasets:
        print(f"\nDownloading {name}...")
        if name == "math500":
            download_math500(args.data_dir)
        elif name == "minerva_math":
            download_minerva_math(args.data_dir)
        elif name == "aime24":
            download_aime24(args.data_dir)
        else:
            print(f"  {name}: not supported by this script (may be private dataset)")
