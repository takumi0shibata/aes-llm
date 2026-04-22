import os
import re
import polars as pl
from sklearn.model_selection import train_test_split
from typing import Optional
import tiktoken


def get_score_range(dataset_name, prompt_id):
    """ASAPデータセットのスコア範囲を取得."""
    score_ranges = {
        "ASAP": {
            1: (2, 12),
            2: (1, 6),
            3: (0, 3),
            4: (0, 3),
            5: (0, 4),
            6: (0, 4),
            7: (0, 30),
            8: (0, 60),
        },
        "TOEFL11": {
            1: (0, 2),
            2: (0, 2),
            3: (0, 2),
            4: (0, 2),
            5: (0, 2),
            6: (0, 2),
            7: (0, 2),
            8: (0, 2),
        },
    }
    return score_ranges[dataset_name][prompt_id]


def _extract_numbers(column):
    return column.map_elements(
        lambda x: re.findall(r"\d+", x)[0], return_dtype=pl.String
    )


def load_toefl_dataset(
    dataset_dir: str, essay_set: Optional[int] = None
) -> pl.DataFrame:
    # Load score data
    test_index = os.path.join(dataset_dir, "data/text/index-test.csv")
    essays = pl.read_csv(
        test_index, new_columns=["essay_id", "essay_set", "original_score"]
    )

    essays = essays.with_columns(
        [
            _extract_numbers(pl.col("essay_id")).cast(pl.Int64).alias("essay_id"),
            _extract_numbers(pl.col("essay_set")).cast(pl.Int64).alias("essay_set"),
        ]
    )

    # Load text data
    text_dir = os.path.join(dataset_dir, "data/text/responses/original")
    data = {"essay_id": [], "essay": []}

    for filename in os.listdir(text_dir):
        if filename.endswith(".txt"):
            essay_id = int(filename.split(".")[0])  # Get essay_id from filename
            with open(os.path.join(text_dir, filename), "r", encoding="utf-8") as file:
                content = file.read()
            data["essay_id"].append(essay_id)
            data["essay"].append(content)

    df = pl.DataFrame(data)

    # Create final dataframe
    essays = essays.join(df, on="essay_id", how="left")
    essays = essays.with_columns(
        pl.when(pl.col("original_score") == "high")
        .then(2)
        .when(pl.col("original_score") == "medium")
        .then(1)
        .otherwise(0)
        .alias("score")
    )

    if essay_set:
        essays = essays.filter(pl.col("essay_set") == essay_set)

    return essays


def load_asap_dataset(
    dataset_dir: str, stratify: bool = False, essay_set: Optional[int] = None
) -> pl.DataFrame:
    data_path = os.path.join(dataset_dir, "training_set_rel3.xlsx")
    df: pl.DataFrame = pl.read_excel(data_path, infer_schema_length=100000)
    df = df.rename({"domain1_score": "score"})
    df = df[["essay_set", "essay_id", "essay", "score"]]
    df = df.drop_nulls("score")

    if essay_set:
        df = df.filter(pl.col("essay_set") == essay_set)

    if not stratify:
        return df
    else:
        score_counts = df.group_by("score").len()
        # Add classes with only one sample directly to test set
        test_df = df.filter(
            pl.col("score").is_in(score_counts.filter(pl.col("len") == 1)["score"])
        )
        df_remaining = df.filter(
            ~pl.col("score").is_in(score_counts.filter(pl.col("len") == 1)["score"])
        )

        # Perform stratified sampling on remaining data
        train_df, tmp_test_df = train_test_split(
            df_remaining,
            test_size=0.1,
            stratify=df_remaining["score"],
            random_state=123,
        )
        test_df = pl.concat([test_df, tmp_test_df], how="vertical")

        return test_df


def get_min_max_scores(prompt: int, attribute: str) -> tuple:
    attribute_ranges = {
        1: {'overall': (2, 12), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6), 'sentence_fluency': (1, 6), 'conventions': (1, 6)},
        2: {'overall': (1, 6), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6), 'sentence_fluency': (1, 6), 'conventions': (1, 6)},
        3: {'overall': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        4: {'overall': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        5: {'overall': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        6: {'overall': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        7: {'overall': (0, 30), 'content': (0, 6), 'organization': (0, 6), 'conventions': (0, 6), 'style': (0, 6)},
        8: {'overall': (0, 60), 'content': (2, 12), 'organization': (2, 12), 'word_choice': (2, 12), 'sentence_fluency': (2, 12), 'conventions': (2, 12), 'voice': (2, 12)},
    }
    
    if prompt in attribute_ranges and attribute in attribute_ranges[prompt]:
        return attribute_ranges[prompt][attribute]
    else:
        raise ValueError(f"Invalid prompt {prompt} or attribute {attribute}.")


def target_attribute(prompt: int) -> list[str]:
    attribute_map = {
        1: ['overall', 'content', 'organization', 'word_choice', 'sentence_fluency', 'conventions'],
        2: ['overall', 'content', 'organization', 'word_choice', 'sentence_fluency', 'conventions'],
        3: ['overall', 'content', 'prompt_adherence', 'language', 'narrativity'],
        4: ['overall', 'content', 'prompt_adherence', 'language', 'narrativity'],
        5: ['overall', 'content', 'prompt_adherence', 'language', 'narrativity'],
        6: ['overall', 'content', 'prompt_adherence', 'language', 'narrativity'],
        7: ['overall', 'content', 'organization', 'conventions', 'style'],
        8: ['overall', 'content', 'organization', 'word_choice', 'sentence_fluency', 'conventions', 'voice'],
    }
    return attribute_map.get(prompt, [])


def count_total_tokens(queries, model="gpt-4o-mini"):
    enc = tiktoken.encoding_for_model(model)
    total_tokens = 0

    for qlist in queries.values():
        for msg in qlist:
            total_tokens += len(enc.encode(msg["content"]))

    return total_tokens