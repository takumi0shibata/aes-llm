import os
import re
import polars as pl
from sklearn.model_selection import train_test_split
from typing import Optional


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
