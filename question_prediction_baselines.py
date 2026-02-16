import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from extract_exam_scores import csv_to_student_response_block
from llm_accuracy import EXAM_SCHEMAS, get_question_slice, truth_pred_metrics_for_filtered_question_df


def load_student_response_df(csv_file):
    text = csv_to_student_response_block(csv_file)
    if text.startswith("Error:"):
        raise ValueError(text)

    rows = []
    for line in text.strip().splitlines():
        if line.strip():
            rows.append(line.replace("||", " ").split())

    if not rows:
        raise ValueError("No student rows found after parsing.")

    return pd.DataFrame(rows)


def slice_question(df, exam, question_num):
    start_col, end_col = get_question_slice(exam, question_num)
    return df.iloc[:, start_col:end_col].copy().reset_index(drop=True)


def train_test_indices(n_rows, test_fraction, seed):
    if n_rows < 2:
        raise ValueError("Need at least 2 rows for train/test split.")
    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_rows)
    rng.shuffle(indices)

    test_n = max(1, int(round(n_rows * test_fraction)))
    train_n = n_rows - test_n
    if train_n < 1:
        test_n -= 1
        train_n = 1

    test_idx = indices[:test_n]
    train_idx = indices[test_n:]
    return train_idx, test_idx


def majority_value(series):
    t_count = int((series == "T").sum())
    f_count = int((series == "F").sum())
    return "T" if t_count >= f_count else "F"


def majority_vector(y_train):
    return [majority_value(y_train[col]) for col in y_train.columns]


def predict_majority(y_train, n_rows):
    maj = majority_vector(y_train)
    arr = np.tile(maj, (n_rows, 1))
    return pd.DataFrame(arr, columns=y_train.columns)


def predict_copy_source(x_test, target_columns):
    target_width = len(target_columns)
    source_width = x_test.shape[1]

    if source_width == target_width:
        arr = x_test.values
    elif source_width == 1:
        arr = np.repeat(x_test.values, target_width, axis=1)
    else:
        # Fallback: use first source subquestion for all target subquestions.
        arr = np.repeat(x_test.iloc[:, [0]].values, target_width, axis=1)

    return pd.DataFrame(arr, columns=target_columns)


def build_conditional_lookup(x_train, y_train):
    buckets = defaultdict(list)
    for i in range(len(x_train)):
        key = tuple(x_train.iloc[i].tolist())
        buckets[key].append(y_train.iloc[i].tolist())

    lookup = {}
    for key, y_rows in buckets.items():
        y_df = pd.DataFrame(y_rows, columns=y_train.columns)
        lookup[key] = majority_vector(y_df)
    return lookup


def predict_conditional_lookup(x_train, y_train, x_test):
    defaults = majority_vector(y_train)
    lookup = build_conditional_lookup(x_train, y_train)

    pred_rows = []
    for i in range(len(x_test)):
        key = tuple(x_test.iloc[i].tolist())
        pred_rows.append(lookup.get(key, defaults))

    return pd.DataFrame(pred_rows, columns=y_train.columns)


def predict_random_weighted(y_train, n_rows, seed):
    rng = np.random.default_rng(seed)
    probs_t = [(y_train[col] == "T").mean() for col in y_train.columns]

    cols = []
    for p in probs_t:
        col = np.where(rng.random(n_rows) < p, "T", "F")
        cols.append(col)

    arr = np.column_stack(cols)
    return pd.DataFrame(arr, columns=y_train.columns)


def run_baselines(csv_file, exam, source_question, target_question, test_fraction, seed):
    df = load_student_response_df(csv_file)

    x_full = slice_question(df, exam, source_question)
    y_full = slice_question(df, exam, target_question)

    train_idx, test_idx = train_test_indices(len(df), test_fraction, seed)
    x_train, x_test = x_full.iloc[train_idx], x_full.iloc[test_idx]
    y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]

    predictions = {
        "majority_target": predict_majority(y_train, len(y_test)),
        "copy_source": predict_copy_source(x_test, y_train.columns),
        "conditional_lookup": predict_conditional_lookup(x_train, y_train, x_test),
        "random_weighted": predict_random_weighted(y_train, len(y_test), seed + 1),
    }

    results = []
    for name, y_pred in predictions.items():
        metrics = truth_pred_metrics_for_filtered_question_df(y_test, y_pred)
        results.append(
            (
                name,
                metrics["accuracy"],
                # metrics["full_question_match_acc"],
                metrics["f1_t"],
                metrics["prop_predict_false_when_false"],
                metrics["prop_predict_true_when_wrong"],
                metrics["prop_llm_predicting_true"],
                metrics["baseline_proportion_true"],
            )
        )

    print(f"CSV: {csv_file}")
    print(f"Exam: {exam}")
    print(f"Predicting q{target_question} from q{source_question}")
    print(f"Rows: total={len(df)}, train={len(train_idx)}, test={len(test_idx)}")
    print("")
    print(
        "Baseline\tAccuracy\tF1(T)\t"
        "PropPredictFalseWhenFalse\tPropPredictTrueWhenWrong\t"
        "PropLLMPredictingTrue\tBaselinePropTrue"
    )
    for (
        name,
        accuracy,
        # full_question_match_acc,
        f1_t,
        prop_predict_false_when_false,
        prop_predict_true_when_wrong,
        prop_llm_predicting_true,
        baseline_proportion_true,
    ) in sorted(results, key=lambda x: x[1], reverse=True):
        print(
            f"{name}\t{accuracy:.4f}\t{f1_t:.4f}\t"
            f"{prop_predict_false_when_false:.4f}\t{prop_predict_true_when_wrong:.4f}\t"
            f"{prop_llm_predicting_true:.4f}\t{baseline_proportion_true:.4f}"
        )


def run_conditional_lookup_source_sweep(csv_file, exam, target_question, test_fraction, seed):
    """
    For one target question, evaluate conditional_lookup using every source question.
    Then print:
    1) best source row (excluding source == target)
    2) averaged metrics across sources (excluding source == target)
    """
    df = load_student_response_df(csv_file)
    num_questions = len(EXAM_SCHEMAS[exam])

    y_full = slice_question(df, exam, target_question)
    train_idx, test_idx = train_test_indices(len(df), test_fraction, seed)
    y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]

    evaluations = []
    for source_question in range(1, num_questions + 1):
        if source_question == target_question:
            continue

        x_full = slice_question(df, exam, source_question)
        x_train, x_test = x_full.iloc[train_idx], x_full.iloc[test_idx]

        y_pred = predict_conditional_lookup(x_train, y_train, x_test)
        metrics = truth_pred_metrics_for_filtered_question_df(y_test, y_pred)

        evaluations.append(
            {
                "source_question": source_question,
                "truth_df": y_test.reset_index(drop=True),
                "pred_df": y_pred.reset_index(drop=True),
                "accuracy": metrics["accuracy"],
                "f1_t": metrics["f1_t"],
                "prop_predict_false_when_false": metrics["prop_predict_false_when_false"],
                "prop_predict_true_when_wrong": metrics["prop_predict_true_when_wrong"],
                "prop_llm_predicting_true": metrics["prop_llm_predicting_true"],
                "baseline_proportion_true": metrics["baseline_proportion_true"],
            }
        )

    if not evaluations:
        raise ValueError("No non-self source questions available.")

    best = max(evaluations, key=lambda row: row["accuracy"])
    metric_keys = [
        "accuracy",
        "f1_t",
        "prop_predict_false_when_false",
        "prop_predict_true_when_wrong",
        "prop_llm_predicting_true",
        "baseline_proportion_true",
    ]
    averages = {
        key: sum(row[key] for row in evaluations) / len(evaluations)
        for key in metric_keys
    }

    print(f"CSV: {csv_file}")
    print(f"Exam: {exam}")
    print(f"Target question: q{target_question}")
    print(f"Rows: total={len(df)}, train={len(train_idx)}, test={len(test_idx)}")
    print("")
    print("Table 1: Best Conditional Lookup Source (excluding target itself)")
    print(
        "SourceQuestion\tAccuracy\tF1(T)\tPropPredictFalseWhenFalse\t"
        "PropPredictTrueWhenWrong\tPropLLMPredictingTrue\tBaselinePropTrue"
    )
    print(
        f"q{best['source_question']}\t{best['accuracy']:.4f}\t{best['f1_t']:.4f}\t"
        f"{best['prop_predict_false_when_false']:.4f}\t{best['prop_predict_true_when_wrong']:.4f}\t"
        f"{best['prop_llm_predicting_true']:.4f}\t{best['baseline_proportion_true']:.4f}"
    )
    print("")
    print("Table 2: Average Conditional Lookup Metrics (excluding target itself)")
    print(
        "Group\tAccuracy\tF1(T)\tPropPredictFalseWhenFalse\t"
        "PropPredictTrueWhenWrong\tPropLLMPredictingTrue\tBaselinePropTrue"
    )
    print(
        f"AverageAcrossSources\t{averages['accuracy']:.4f}\t{averages['f1_t']:.4f}\t"
        f"{averages['prop_predict_false_when_false']:.4f}\t"
        f"{averages['prop_predict_true_when_wrong']:.4f}\t"
        f"{averages['prop_llm_predicting_true']:.4f}\t"
        f"{averages['baseline_proportion_true']:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple baseline predictors for question-to-question accuracy."
    )
    parser.add_argument(
        "--csv-file",
        default="llm_data/exam1/exam1_crawford.csv",
        help="Path to student response CSV.",
    )
    parser.add_argument(
        "--exam",
        default="exam1",
        choices=["exam1", "exam2", "final"],
        help="Exam schema to use.",
    )
    parser.add_argument(
        "--source-question",
        type=int,
        default=9,
        help="Source question number (1-based).",
    )
    parser.add_argument(
        "--target-question",
        type=int,
        default=1,
        help="Target question number (1-based).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.3,
        help="Test split fraction between 0 and 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split and random baseline.",
    )
    parser.add_argument(
        "--sweep-conditional-lookup",
        action="store_true",
        help="Evaluate conditional_lookup over all source questions for one target question.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.sweep_conditional_lookup:
        run_conditional_lookup_source_sweep(
            csv_file=args.csv_file,
            exam=args.exam,
            target_question=args.target_question,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
    else:
        run_baselines(
            csv_file=args.csv_file,
            exam=args.exam,
            source_question=args.source_question,
            target_question=args.target_question,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
