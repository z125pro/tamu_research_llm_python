import pandas as pd
from extract_exam_scores import csv_to_student_response_block

EXAM1_BLOCK_SIZES = (2, 1, 5, 1, 2, 3, 1, 1, 1, 1)
EXAM2_BLOCK_SIZES = (2, 1, 1, 1, 2, 1, 8, 12, 4, 6)
FINAL_BLOCK_SIZES = (
    1, 4, 3, 3, 1, 1, 2, 4, 4,
    1, 1, 2, 8, 9, 6, 1, 12, 4
)
EXAM_SCHEMAS = {
    "exam1": EXAM1_BLOCK_SIZES,
    "exam2": EXAM2_BLOCK_SIZES,
    "final": FINAL_BLOCK_SIZES,
}

# def majority_vote(tokens, default="T"):
#     if not tokens:
#         return default
#     return "T" if tokens.count("T") >= tokens.count("F") else "F"


# def normalize_line(line, block_sizes, global_default="T"):
#     blocks = [b.strip() for b in line.split("||")]

#     if len(blocks) != len(block_sizes):
#         raise ValueError(
#             f"Wrong number of blocks: {len(blocks)} (expected {len(block_sizes)})"
#         )

#     normalized_blocks = []

#     for block, expected_size in zip(blocks, block_sizes):
#         tokens = block.split()

#         if len(tokens) > expected_size:
#             raise ValueError(f"Too many tokens in block: {tokens}")

#         fill_value = majority_vote(tokens, global_default)
#         tokens += [fill_value] * (expected_size - len(tokens))

#         normalized_blocks.append(" ".join(tokens))

#     return "||".join(normalized_blocks)


# def parse_responses_to_df(
#     responses,
#     normalize=False,
#     exam="exam1",
#     global_default="T"
# ):
#     """
#     Parses T/F responses into a DataFrame.

#     If normalize=True:
#         - Enforces exam block structure
#         - Imputes missing elements (LLM output)
#     If normalize=False:
#         - Parses as-is
#     """

#     rows = []
#     lines = responses.strip().split("\n")

#     block_sizes = EXAM_SCHEMAS.get(exam)

#     for line_num, line in enumerate(lines, start=1):
#         if not line.strip():
#             continue

#         if normalize:
#             if block_sizes is None:
#                 raise ValueError(f"No schema found for exam='{exam}'")

#             try:
#                 line = normalize_line(
#                     line,
#                     block_sizes=block_sizes,
#                     global_default=global_default
#                 )
#             except Exception as e:
#                 raise RuntimeError(f"Line {line_num} failed normalization: {e}")

#         # Common parsing logic (shared)
#         flat_tokens = line.replace("||", " ").split()
#         rows.append(flat_tokens)

#     return pd.DataFrame(rows)


def check_schema_match(line, block_sizes):
    """
    Returns True if the line strictly follows the block structure 
    (correct number of blocks, correct token count per block).
    Returns False otherwise.
    """
    blocks = [b.strip() for b in line.split("||")]

    # 1. Check number of blocks
    if len(blocks) != len(block_sizes):
        return False

    # 2. Check each block size
    for block, expected_size in zip(blocks, block_sizes):
        tokens = block.split()
        if len(tokens) != expected_size: # Strict check: must equal exactly
            return False

    return True

def load_aligned_token_dfs(original_file, llm_file, exam="exam1"):
    """
    Returns aligned original and LLM token DataFrames after
    length truncation and malformed LLM row filtering.
    """
    original_text = csv_to_student_response_block(original_file)
    if "Error:" in original_text:
        raise ValueError(original_text)

    try:
        with open(llm_file, "r", encoding="utf-8") as f:
            llm_text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find text file: {llm_file}")
    lines_orig = [line.strip() for line in original_text.strip().splitlines() if line.strip()]
    lines_llm_raw = [line.strip() for line in llm_text.strip().splitlines() if line.strip()]

    lines_llm = []
    removed_header_rows = 0
    for line in lines_llm_raw:
        if line.lower().startswith("student responses"):
            removed_header_rows += 1
            continue
        lines_llm.append(line)

    if removed_header_rows:
        print(f"Info: Removed {removed_header_rows} header row(s) from LLM output.")

    if len(lines_orig) != len(lines_llm):
        min_len = min(len(lines_orig), len(lines_llm))
        print(f"Warning: File length mismatch ({len(lines_orig)} vs {len(lines_llm)}). Truncating to {min_len} lines.")
        lines_orig = lines_orig[:min_len]
        lines_llm = lines_llm[:min_len]

    block_sizes = EXAM_SCHEMAS.get(exam)
    if not block_sizes:
        raise ValueError(f"Unknown exam schema: {exam}")

    valid_rows_orig = []
    valid_rows_llm = []
    dropped_indices = []

    for idx, (l_orig, l_llm) in enumerate(zip(lines_orig, lines_llm)):
        if check_schema_match(l_llm, block_sizes):
            valid_rows_orig.append(l_orig.replace("||", " ").split())
            valid_rows_llm.append(l_llm.replace("||", " ").split())
        else:
            dropped_indices.append(idx)

    if dropped_indices:
        print(f"Dropped {len(dropped_indices)} malformed rows from comparison.")

    df_orig = pd.DataFrame(valid_rows_orig)
    df_llm = pd.DataFrame(valid_rows_llm)

    if df_orig.empty:
        raise ValueError("No valid rows remaining after filtering.")

    if df_orig.shape != df_llm.shape:
        raise ValueError(f"Shape mismatch: {df_orig.shape} vs {df_llm.shape}")

    return df_orig, df_llm

def compare_student_files(original_file, llm_file, exam="exam1", question_num=None):
    """
    Compares the original student responses with the LLM-generated responses.
    Returns a DataFrame of 1s (Match) and 0s (Mismatch).
    If question_num is provided, returns only that question's columns.
    """
    df_orig, df_llm = load_aligned_token_dfs(original_file, llm_file, exam=exam)

    # Compare
    comparison_values = (df_orig.values == df_llm.values).astype(int)
    # Using index/columns from df_orig preserves structure if you need to export it later
    comparison_df = pd.DataFrame(comparison_values, index=df_orig.index, columns=df_orig.columns)

    if question_num is not None:
        start_col, end_col = get_question_slice(exam, question_num)
        comparison_df = comparison_df.iloc[:, start_col:end_col]

    return comparison_df

def proportion_correct(binary_df):
    """
    Computes proportion of 1s in a 0/1 DataFrame.
    """
    return binary_df.values.mean()

def truth_pred_metrics_for_filtered_question_df(truth_df, pred_df):
    """
    Computes metrics for already-filtered truth/prediction DataFrames that
    represent a single question block (or any aligned token block).
    """
    if truth_df.shape != pred_df.shape:
        raise ValueError(f"Shape mismatch: {truth_df.shape} vs {pred_df.shape}")

    truth = truth_df.values
    pred = pred_df.values

    false_positives = int(((pred == "T") & (truth == "F")).sum())
    false_negatives = int(((pred == "F") & (truth == "T")).sum())
    true_positives = int(((pred == "T") & (truth == "T")).sum())
    true_t_count = int((truth == "T").sum())
    true_f_count = int((truth == "F").sum())
    total_count = true_t_count + true_f_count

    precision_den = true_positives + false_positives
    recall_den = true_positives + false_negatives
    precision = true_positives / precision_den if precision_den > 0 else 0.0
    recall = true_positives / recall_den if recall_den > 0 else 0.0
    f1_t = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    correct_mask = (pred == truth)
    wrong_mask = ~correct_mask
    correct_total = int(correct_mask.sum())
    wrong_total = int(wrong_mask.sum())
    pred_false_when_false = int(((pred == "F") & (truth == "F")).sum())
    pred_true_when_wrong = int(((pred == "T") & wrong_mask).sum())
    llm_true_count = int((pred == "T").sum())

    prop_recall_false = (pred_false_when_false / true_f_count) if true_f_count > 0 else 0.0
    prop_recall_true = recall
    prop_true_when_wrong = (pred_true_when_wrong / wrong_total) if wrong_total > 0 else 0.0
    prop_llm_predicting_true = (llm_true_count / total_count) if total_count > 0 else 0.0
    accuracy = (correct_total / total_count) if total_count > 0 else 0.0
    baseline_proportion_true = (true_t_count / total_count) if total_count > 0 else 0.0
    full_question_match_acc = float(correct_mask.all(axis=1).mean()) if correct_mask.size > 0 else 0.0

    return {
        "accuracy": accuracy,
        "f1_t": f1_t,
        "prop_recall_false": prop_recall_false,
        "prop_recall_true": prop_recall_true,
        "prop_true_when_wrong": prop_true_when_wrong,
        "prop_llm_predicting_true": prop_llm_predicting_true,
        "baseline_proportion_true": baseline_proportion_true,
        "full_question_match_acc": full_question_match_acc,
    }

def get_question_slice(exam, question_num):
    """
    Returns [start_col, end_col) for a 1-based question number using exam block sizes.
    """
    block_sizes = EXAM_SCHEMAS.get(exam)
    if not block_sizes:
        raise ValueError(f"Unknown exam schema: {exam}")

    if question_num < 1 or question_num > len(block_sizes):
        raise ValueError(
            f"Question {question_num} is out of range for {exam}. "
            f"Valid range: 1 to {len(block_sizes)}"
        )

    start_col = sum(block_sizes[:question_num - 1])
    end_col = start_col + block_sizes[question_num - 1]
    return start_col, end_col

def f1_and_truth_proportion_metrics_by_question(
    original_file,
    llm_file,
    question_nums=None,
    exam="exam1",
    return_averages=False,
    shuffle_llm_rows=False,
    shuffle_seed=None,
    bootstrap_repeats=1,
    bootstrap_seed=None,
):
    """
    Flexible metrics helper.

    Supported modes:
    1) Single file, all questions (backward-compatible):
       llm_file: str, question_nums=None
    2) Single file, selected questions:
       llm_file: str, question_nums=[...]
    3) Pairwise file/question evaluation:
       llm_file: [file1, file2, ...], question_nums=[q1, q2, ...]
       (same length arrays)

    4) Permutation baseline mode:
       shuffle_llm_rows=True to shuffle LLM rows before scoring.
       Optionally set shuffle_seed for reproducible shuffles.

    5) Bootstrap expansion mode:
       bootstrap_repeats > 1 to append repeated resamples of df_llm
       (with replacement), paired with repeated df_orig blocks.
       Optionally set bootstrap_seed for reproducible resamples.

    Returns per-row metrics with:
    - accuracy
    - f1_t
    - prop_recall_false
    - prop_recall_true
    - prop_true_when_wrong
    - prop_llm_predicting_true
    - baseline_proportion_true
    """
    block_sizes = EXAM_SCHEMAS[exam]

    if isinstance(llm_file, (list, tuple)):
        llm_files = list(llm_file)
    else:
        llm_files = [llm_file]

    if question_nums is None:
        if len(llm_files) == 1:
            question_nums = list(range(1, len(block_sizes) + 1))
        else:
            raise ValueError(
                "question_nums is required when passing multiple llm files."
            )
    elif isinstance(question_nums, int):
        question_nums = [question_nums]
    else:
        question_nums = list(question_nums)

    # Pairing rules:
    # - 1 file + many questions -> reuse same file
    # - N files + N questions -> pair by index
    if len(llm_files) == 1 and len(question_nums) >= 1:
        eval_pairs = [(llm_files[0], q) for q in question_nums]
    elif len(llm_files) == len(question_nums):
        eval_pairs = list(zip(llm_files, question_nums))
    else:
        raise ValueError(
            "Length mismatch: provide either one llm file for many questions, "
            "or equal-length llm_file/question_nums arrays."
        )

    # Cache tokenized DataFrames so repeated files are loaded once.
    file_cache = {}
    results = []

    for llm_path, question_num in eval_pairs:
        if llm_path not in file_cache:
            df_orig, df_llm = load_aligned_token_dfs(
                original_file, llm_path, exam=exam
            )
            if shuffle_llm_rows:
                # Permutation baseline: keep truth fixed, break row alignment in predictions.
                row_state = None
                if shuffle_seed is not None:
                    row_state = shuffle_seed + (sum(ord(ch) for ch in str(llm_path)) % 100000)
                df_orig = df_orig.reset_index(drop=True)
                df_llm = df_llm.sample(frac=1, random_state=row_state).reset_index(drop=True)

            repeats = 1 if bootstrap_repeats is None else int(bootstrap_repeats)
            if repeats < 1:
                raise ValueError("bootstrap_repeats must be >= 1")
            if repeats > 1:
                df_orig_base = df_orig.reset_index(drop=True)
                df_llm_base = df_llm.reset_index(drop=True)

                # Expand both truth and prediction pools, then resample predictions.
                df_orig = pd.concat([df_orig_base] * repeats, ignore_index=True)
                df_llm_pool = pd.concat([df_llm_base] * repeats, ignore_index=True)

                row_state = None
                if bootstrap_seed is not None:
                    row_state = bootstrap_seed + (sum(ord(ch) for ch in str(llm_path)) % 100000)

                df_llm = df_llm_pool.sample(
                    n=len(df_llm_pool),
                    replace=True,
                    random_state=row_state,
                ).reset_index(drop=True)

            file_cache[llm_path] = (df_orig, df_llm)
        df_orig, df_llm = file_cache[llm_path]

        start_col, end_col = get_question_slice(exam, question_num)
        truth_df = df_orig.iloc[:, start_col:end_col]
        pred_df = df_llm.iloc[:, start_col:end_col]
        metrics = truth_pred_metrics_for_filtered_question_df(truth_df, pred_df)

        results.append(
            {
                "llm_file": llm_path,
                "question": question_num,
                "accuracy": metrics["accuracy"],
                "f1_t": metrics["f1_t"],
                "prop_recall_false": metrics["prop_recall_false"],
                "prop_recall_true": metrics["prop_recall_true"],
                "prop_true_when_wrong": metrics["prop_true_when_wrong"],
                "prop_llm_predicting_true": metrics["prop_llm_predicting_true"],
                "baseline_proportion_true": metrics["baseline_proportion_true"],
            }
        )

    if not return_averages:
        return results

    metric_keys = [
        "accuracy",
        "f1_t",
        "prop_recall_false",
        "prop_recall_true",
        "prop_true_when_wrong",
        "prop_llm_predicting_true",
        "baseline_proportion_true",
    ]
    averages = {
        key: sum(row[key] for row in results) / len(results) if results else 0.0
        for key in metric_keys
    }
    return results, averages
if __name__ == "__main__":
    original_file = "llm_data/exam1/exam1_crawford.csv"
    llm_file = "results_gemini_flash_1_exam1_crawford_masked_q10.txt"

    comparison_df = compare_student_files(original_file, llm_file, exam = "exam1")

    accuracy = proportion_correct(comparison_df)
    print(f"Proportion correct: {accuracy:.4f}")

    metric_rows, metric_averages = f1_and_truth_proportion_metrics_by_question(
        original_file,
        llm_file,
        exam="exam1",
        return_averages=True,
    )
    print(
        "Question\tAccuracy\tF1(T)\tPropRecallFalse\tPropRecallTrue\t"
        "PropTrueWhenWrong\tPropLLMPredictingTrue\tBaselinePropTrue"
    )
    for row in metric_rows:
        print(
            f"q{row['question']}\t{row['accuracy']:.4f}\t{row['f1_t']:.4f}\t"
            f"{row['prop_recall_false']:.4f}\t{row['prop_recall_true']:.4f}\t"
            f"{row['prop_true_when_wrong']:.4f}\t{row['prop_llm_predicting_true']:.4f}\t"
            f"{row['baseline_proportion_true']:.4f}"
        )
    print(
        f"Average\t{metric_averages['accuracy']:.4f}\t{metric_averages['f1_t']:.4f}\t"
        f"{metric_averages['prop_recall_false']:.4f}\t"
        f"{metric_averages['prop_recall_true']:.4f}\t"
        f"{metric_averages['prop_true_when_wrong']:.4f}\t{metric_averages['prop_llm_predicting_true']:.4f}\t"
        f"{metric_averages['baseline_proportion_true']:.4f}"
    )





