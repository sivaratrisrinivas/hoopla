import argparse
import json
from pathlib import Path

from lib.evaluation import evaluate_command, format_markdown_table


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write the full JSON report to this path",
    )

    args = parser.parse_args()
    limit = args.limit
    report = evaluate_command(limit)

    print(f"k={limit}")
    print(f"date={report['date']}")
    print(f"queries={report['dataset']['num_queries']}")
    print(f"documents={report['dataset']['num_documents']}")
    print(f"embedding_model={report['models']['embedding']}")
    print(f"cross_encoder_model={report['models']['cross_encoder']}")
    print()
    print(format_markdown_table(report))
    print()

    for config_name, cfg in report["configurations"].items():
        print(f"## {cfg['label']}")
        if cfg["failures"]:
            for item in cfg["failures"]:
                print(f"- FAILED {item['query']}: {item['error']}")
        for query, res in cfg["per_query"].items():
            if res.get("error"):
                print(f"- Query: {query}")
                print(f"  - Error: {res['error']}")
                continue
            print(f"- Query: {query}")
            print(f"  - Precision@{limit}: {res['precision']:.4f}")
            print(f"  - Recall@{limit}: {res['recall']:.4f}")
            print(f"  - F1 Score: {res['f1_score']:.4f}")
            print(f"  - Latency ms: {res['latency_ms']:.1f}")
            print(f"  - Retrieved: {', '.join(res['retrieved'])}")
            print(f"  - Relevant: {', '.join(res['relevant'])}")
            print()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: v for k, v in report.items()}
        output_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False) + "\n")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
