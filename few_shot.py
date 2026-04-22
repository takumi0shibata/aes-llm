from pathlib import Path
import argparse
import json

from dotenv import load_dotenv
from tqdm import tqdm
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from json_repair import repair_json

from tokenrail import BatchExecutor, PerRequestJsonSink, RailClient, ResultsJsonlSink, RollingMetricsMonitor
from tokenrail.executor import batch_items_from_queries

from utils import get_min_max_scores, target_attribute, count_total_tokens

load_dotenv()


def main(prompt_id, att, args):
    output_dir = Path(args.out_dir) / args.model.split('/')[-1] / str(prompt_id) / att
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "results.jsonl"

    # Load dataset and split into few-shot / test
    asap = pl.read_csv(args.db_path, infer_schema_length=20000)
    asap = asap.filter(pl.col("essay_set") == prompt_id)
    asap = asap.select(["essay_id", "essay", att])

    few_shot_df, test_df = train_test_split(
        asap,
        test_size=len(asap) - args.n_few,
        random_state=args.seed,
        shuffle=True,
    )
    print(f"few_shot_df len: {len(few_shot_df)}")
    print(f"test_df len: {len(test_df)}")

    # Build few-shot examples (sorted by score ascending)
    minscore, maxscore = get_min_max_scores(prompt_id, att)
    many_shot_samples = ""
    for essay, score in few_shot_df.sort(by=att).select(['essay', att]).iter_rows():
        many_shot_samples += f"score: {score}\nessay text: {essay}\n"

    # Load prompt templates
    prompt_dir = Path(args.llm_prompt_dir)
    user_message = (prompt_dir / "few_shot_user.md").read_text()
    essay_prompt = (prompt_dir / "asap_original" / f"prompt_{prompt_id}.md").read_text()
    rubric = next((prompt_dir / "original_trait" / att).glob(f"*_{prompt_id}*")).read_text()

    # Build queries
    queries = {}
    for essay_id, essay in tqdm(test_df.sort(by='essay_id').select(['essay_id', 'essay']).iter_rows()):
        message = (
            user_message
            .replace('{prompt}', essay_prompt)
            .replace('{rubric}', rubric)
            .replace('{examples}', many_shot_samples)
            .replace('{minscore}', str(minscore))
            .replace('{maxscore}', str(maxscore))
            .replace('{essay}', essay)
        )
        queries[essay_id] = [{'role': 'user', 'content': message}]

    print(f"総トークン数: {count_total_tokens(queries):,}")

    # Run inference
    client = RailClient.vllm(
        model_id=args.model,
        family=args.model_family,
        batch_flush_size=256,
        dtype="bfloat16",
        max_model_len=32000,
        gpu_memory_utilization=0.92,
        enable_prefix_caching=True,
        trust_remote_code=True,
        seed=args.seed,
    )

    items = batch_items_from_queries(
        queries,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        enable_thinking=False,
    )

    executor = BatchExecutor(
        client=client,
        sinks=[
            ResultsJsonlSink(jsonl_path),
            PerRequestJsonSink(output_dir / "requests"),
        ],
        monitor=RollingMetricsMonitor(),
    )
    stats = executor.run(items)
    summary = stats.to_dict()

    # Parse results
    results = []
    errors = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            entry_id = entry['id']
            text = entry.get("output_text", "")

            try:
                score = json.loads(repair_json(text))['score']
                results.append({"essay_id": int(entry_id), "pred_score": score})
            except (json.JSONDecodeError, KeyError, TypeError):
                results.append({"essay_id": int(entry_id), "pred_score": None})
                errors.append(f"ID {entry_id} のスコアが不正な形式です: {text}")

    print(f"抽出結果: {len(results)}件、エラー: {len(errors)}件")

    # Compute QWK
    results_df = pl.DataFrame(results)
    final_df = test_df.join(results_df, on='essay_id', how='left')

    n_before = len(final_df)
    final_df = final_df.drop_nulls()
    n_dropped = n_before - len(final_df)
    if n_dropped > 0:
        print(f"警告: pred_score が null の {n_dropped} 件を除外しました。")

    qwk = cohen_kappa_score(
        final_df["pred_score"].to_numpy(),
        final_df[att].to_numpy(),
        weights="quadratic",
        labels=list(range(minscore, maxscore + 1)),
    )

    summary['decode_error'] = len(errors)
    summary['n_dropped'] = n_dropped
    summary['n_evaluated'] = len(final_df)
    summary['qwk'] = qwk
    summary |= vars(args)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["Qwen/Qwen3.5-9B", "google/gemma-4-e4b-it"])
    parser.add_argument("--model-family", type=str, required=True, choices=["qwen", "gemma"])
    parser.add_argument("--db-path", type=str, default="./dataset/asap_with_traits.csv")
    parser.add_argument("--llm-prompt-dir", type=str, default="./llm_prompts/")
    parser.add_argument("--out-dir", type=str, default="./out/few_shot/")
    parser.add_argument("--n-few", type=int, default=30)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    args = parser.parse_args()

    for prompt_id in range(1, 9):
        for att in target_attribute(prompt_id):
            print(f"====PROMPT: {prompt_id}, ATT: {att} ===")
            main(prompt_id, att, args)