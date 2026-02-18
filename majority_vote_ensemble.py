import argparse
import random
from pathlib import Path

from extract_exam_scores import csv_to_student_response_block
from llm_accuracy import EXAM_SCHEMAS, check_schema_match


DEFAULT_MODEL_DIRS = {
    "chatgpt": "results/v1.1/gpt_5_2",
    "gemini_flash": "results/v1.1/gemini_flash",
    "gemini_pro": "results/v1.1/gemini_pro",
    "claude": "results/v1.1/claude_opus_4_6",
    "grok": "results/v1.1/grok_4",
    "zai": "results/v1.1/zai_glm5",
}


def clean_llm_lines(file_path: Path):
    text = file_path.read_text(encoding="utf-8")
    raw = [line.strip() for line in text.splitlines() if line.strip()]
    return [line for line in raw if not line.lower().startswith("student responses")]


def parse_blocks(line: str):
    return [block.strip().split() for block in line.strip().split("||")]


def pick_model_file(model_dir: Path, question_num: int):
    candidates = sorted(model_dir.glob(f"*masked_q{question_num}.txt"))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Prefer newest when multiple candidates exist.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def majority_token(votes, rng):
    t_count = votes.count("T")
    f_count = votes.count("F")
    if t_count > f_count:
        return "T"
    if f_count > t_count:
        return "F"
    return rng.choice(["T", "F"])


def build_majority_vote_file(
    original_lines,
    question_num,
    block_sizes,
    model_line_map,
    seed,
):
    q_idx = question_num - 1
    q_width = block_sizes[q_idx]
    rng = random.Random(seed + question_num)

    out_lines = []
    malformed_by_model = {name: 0 for name in model_line_map}
    missing_by_model = {name: 0 for name in model_line_map}

    for row_idx, orig_line in enumerate(original_lines):
        orig_blocks = parse_blocks(orig_line)
        votes_by_token = [[] for _ in range(q_width)]

        for model_name, llm_lines in model_line_map.items():
            if row_idx >= len(llm_lines):
                missing_by_model[model_name] += 1
                continue

            llm_line = llm_lines[row_idx]
            if not check_schema_match(llm_line, block_sizes):
                malformed_by_model[model_name] += 1
                continue

            llm_blocks = parse_blocks(llm_line)
            q_tokens = llm_blocks[q_idx]

            if len(q_tokens) != q_width:
                malformed_by_model[model_name] += 1
                continue

            for tok_idx, tok in enumerate(q_tokens):
                if tok in ("T", "F"):
                    votes_by_token[tok_idx].append(tok)

        chosen_tokens = []
        for tok_idx in range(q_width):
            votes = votes_by_token[tok_idx]
            if votes:
                chosen_tokens.append(majority_token(votes, rng))
            else:
                chosen_tokens.append(rng.choice(["T", "F"]))

        orig_blocks[q_idx] = chosen_tokens
        out_lines.append("||".join(" ".join(block) for block in orig_blocks))

    return out_lines, malformed_by_model, missing_by_model


def build_all_majority_vote_files(
    original_file,
    exam,
    output_dir,
    question_nums,
    model_dirs,
    seed,
):
    block_sizes = EXAM_SCHEMAS.get(exam)
    if not block_sizes:
        raise ValueError(f"Unknown exam schema: {exam}")

    original_text = csv_to_student_response_block(original_file)
    if original_text.startswith("Error:"):
        raise ValueError(original_text)

    original_lines = [line.strip() for line in original_text.splitlines() if line.strip()]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Students in original file: {len(original_lines)}")
    print(f"Writing ensemble files to: {output_dir}")

    model_dir_paths = {name: Path(path) for name, path in model_dirs.items()}

    for q in question_nums:
        model_line_map = {}
        used_files = {}

        for model_name, model_dir in model_dir_paths.items():
            model_file = pick_model_file(model_dir, q)
            if model_file is None:
                continue
            model_line_map[model_name] = clean_llm_lines(model_file)
            used_files[model_name] = model_file

        if not model_line_map:
            print(f"q{q}: skipped (no model files found)")
            continue

        out_lines, malformed_by_model, missing_by_model = build_majority_vote_file(
            original_lines=original_lines,
            question_num=q,
            block_sizes=block_sizes,
            model_line_map=model_line_map,
            seed=seed,
        )

        out_file = output_dir / f"results_majority_vote_{exam}_masked_q{q}.txt"
        out_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

        used = ", ".join(f"{name}:{used_files[name].name}" for name in sorted(used_files))
        malformed_total = sum(malformed_by_model.values())
        missing_total = sum(missing_by_model.values())

        print(
            f"q{q}: wrote {out_file.name} | models={len(model_line_map)} "
            f"| malformed_rows={malformed_total} | missing_rows={missing_total}"
        )
        print(f"    files: {used}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build masked-question majority-vote ensemble files across model outputs."
    )
    parser.add_argument("--original-file", default="llm_data/exam1/exam1_crawford.csv")
    parser.add_argument("--exam", default="exam1", choices=list(EXAM_SCHEMAS.keys()))
    parser.add_argument("--output-dir", default="results/v1.1/majority_vote")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--questions", type=int, nargs="+", default=list(range(1, 11)))
    return parser.parse_args()


def main():
    args = parse_args()
    build_all_majority_vote_files(
        original_file=args.original_file,
        exam=args.exam,
        output_dir=args.output_dir,
        question_nums=args.questions,
        model_dirs=DEFAULT_MODEL_DIRS,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
