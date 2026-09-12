"""Evaluate one collection with a small grounded question set using Ragas."""

from pathlib import Path

import pandas as pd

from src.rag_chain import answer_question

COLLECTION = "novel"
QUESTIONS = [
    {
        "question": "Who arrived at the old mansion when the storm broke?",
        "ground_truth": "Inspector Rao arrived at the old mansion.",
    },
    {
        "question": "What had vanished from the study?",
        "ground_truth": "The master's ledger had vanished from the study.",
    },
    {
        "question": "What did Rao notice leading away from the study door?",
        "ground_truth": "He noticed muddy footprints leading away from the study door.",
    },
    {
        "question": "What was inside the open desk drawer?",
        "ground_truth": "It was empty except for a torn corner of a page.",
    },
    {
        "question": "Who might have known where the ledger was kept?",
        "ground_truth": "The master's nephew and perhaps the accountant who visited last Tuesday.",
    },
]


def collect_examples() -> list[dict[str, object]]:
    examples = []
    for item in QUESTIONS:
        result = answer_question(item["question"], COLLECTION)
        examples.append(
            {
                "question": item["question"],
                "answer": result["answer"],
                "retrieved_contexts": [source["text"] for source in result["sources"]],
                "ground_truth": item["ground_truth"],
            }
        )
    return examples


def evaluate() -> pd.DataFrame:
    try:
        from datasets import Dataset
        from ragas import evaluate as run_ragas
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as error:
        raise RuntimeError(
            "Ragas could not be imported. Install a Ragas version compatible "
            "with the installed LangChain packages."
        ) from error

    dataset = Dataset.from_list(collect_examples())
    result = run_ragas(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    frame = result.to_pandas() if hasattr(result, "to_pandas") else pd.DataFrame(result)
    output_path = Path(__file__).resolve().parents[1] / "outputs" / "eval_results.csv"
    frame.to_csv(output_path, index=False)
    print(frame)
    print(f"Evaluation results saved to {output_path}")
    return frame


if __name__ == "__main__":
    evaluate()
