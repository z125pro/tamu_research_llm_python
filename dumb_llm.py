import argparse
import logging
import os
import random
import sys
import time

import anthropic
from google import genai
from google.genai import types
from openai import OpenAI
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_random_exponential

from extract_exam_scores import csv_to_student_response_block

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

robust_retry = retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.INFO),
)

PROFILE_SETTINGS = {
    "very_low": {
        "reasoning_effort": "low",
        "thinking_budget_tokens": 1024,
        "max_output_tokens": 1200,
        "temperature": 1.2,
        "top_p": 1.0,
        "corruption_rate": 0.12,
    },
    "low": {
        "reasoning_effort": "low",
        "thinking_budget_tokens": 2048,
        "max_output_tokens": 1800,
        "temperature": 0.8,
        "top_p": 1.0,
        "corruption_rate": 0.06,
    },
    "medium_low": {
        "reasoning_effort": "low",
        "thinking_budget_tokens": 4096,
        "max_output_tokens": 2600,
        "temperature": 0.4,
        "top_p": 1.0,
        "corruption_rate": 0.0,
    },
}

LOW_EFFORT_OVERLAY = """
LOW-PERFORMANCE MODE:
- Simulate a weak, error-prone student profile.
- Use shallow heuristics, not deep chain-of-thought.
- If uncertain, make a quick best guess rather than extended reasoning.
- Keep outputs strictly in required T/F structure.
""".strip()


def merged_settings(
    profile,
    max_output_tokens=None,
    reasoning_effort=None,
    thinking_budget_tokens=None,
    temperature=None,
    top_p=None,
    corruption_rate=None,
):
    if profile not in PROFILE_SETTINGS:
        raise ValueError(f"Unknown profile '{profile}'. Valid: {list(PROFILE_SETTINGS)}")

    settings = dict(PROFILE_SETTINGS[profile])
    overrides = {
        "max_output_tokens": max_output_tokens,
        "reasoning_effort": reasoning_effort,
        "thinking_budget_tokens": thinking_budget_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "corruption_rate": corruption_rate,
    }
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
    return settings


def with_low_effort_overlay(system_prompt):
    return f"{system_prompt.strip()}\n\n{LOW_EFFORT_OVERLAY}"


@robust_retry
def call_chatgpt(system_prompt, student_prompt, variant="mini", settings=None):
    settings = settings or PROFILE_SETTINGS["very_low"]
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if variant == "mini":
        model_name = "gpt-5-mini"
    elif variant in {"5.2", "full"}:
        model_name = "gpt-5.2"
    else:
        model_name = variant

    response = client.responses.create(
        model=model_name,
        reasoning={"effort": settings["reasoning_effort"]},
        max_output_tokens=settings["max_output_tokens"],
        instructions=system_prompt,
        input=[
            {
                "role": "user",
                "content": student_prompt,
            }
        ],
    )
    return response.output_text.strip()


@robust_retry
def call_grok(system_prompt, student_prompt, variant="fast-reasoning", settings=None):
    from xai_sdk import Client
    from xai_sdk.chat import system, user

    settings = settings or PROFILE_SETTINGS["very_low"]
    client = Client(api_key=os.getenv("XAI_API_KEY"))

    if variant == "fast-reasoning":
        model_name = "grok-4-1-fast-reasoning"
    elif variant in {"4", "latest"}:
        model_name = "grok-4-latest"
    else:
        model_name = variant

    chat = client.chat.create(
        model=model_name,
        store_messages=False,
        temperature=settings["temperature"],
    )
    chat.append(system(system_prompt))
    chat.append(user(student_prompt))
    response = chat.sample()
    return response.content.strip()


@robust_retry
def call_deepseek(system_prompt, student_prompt, variant="v3.2", settings=None):
    settings = settings or PROFILE_SETTINGS["very_low"]
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    if variant.startswith("deepseek/"):
        model_name = variant
    else:
        model_name = f"deepseek/deepseek-{variant}"

    response = client.chat.completions.create(
        model=model_name,
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        max_tokens=settings["max_output_tokens"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": student_prompt},
        ],
        extra_body={
            "reasoning": {
                "effort": settings["reasoning_effort"],
                "exclude": True,
            }
        },
    )

    return response.choices[0].message.content.strip()


@robust_retry
def call_claude(system_prompt, student_prompt, variant="haiku", settings=None):
    settings = settings or PROFILE_SETTINGS["very_low"]
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    if variant in {"haiku", "sonnet", "opus"}:
        model_name = f"claude-{variant}-4-5"
    else:
        model_name = variant

    response = client.messages.create(
        model=model_name,
        max_tokens=settings["max_output_tokens"],
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": student_prompt,
            }
        ],
        thinking={
            "type": "enabled",
            "budget_tokens": settings["thinking_budget_tokens"],
        },
    )

    for block in response.content:
        if block.type == "text":
            return block.text.strip()
    return ""


@robust_retry
def call_gemini(system_prompt, student_prompt, variant="flash", settings=None):
    settings = settings or PROFILE_SETTINGS["very_low"]
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options={"timeout": 120000},
    )

    if variant == "flash":
        model_name = "gemini-3-flash-preview"
    elif variant in {"pro", "3-pro"}:
        model_name = "gemini-3-pro-preview"
    else:
        model_name = variant

    thinking_kwargs = {"includeThoughts": False}
    if settings.get("thinking_budget_tokens") is not None:
        thinking_kwargs["thinkingBudget"] = settings["thinking_budget_tokens"]

    try:
        thinking_config = types.ThinkingConfig(**thinking_kwargs)
    except TypeError:
        thinking_config = types.ThinkingConfig(includeThoughts=False)

    response = client.models.generate_content(
        model=model_name,
        contents=[student_prompt],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            max_output_tokens=settings["max_output_tokens"],
            thinking_config=thinking_config,
        ),
    )

    return response.text.strip()


def looks_like_tf_output(text):
    allowed = set("TFX| \t\r\n")
    return all(ch in allowed for ch in text)


def flip_tf_tokens(text, flip_rate, seed):
    if flip_rate <= 0:
        return text
    if not looks_like_tf_output(text):
        logger.warning("Skipping corruption because output is not strict T/F format.")
        return text

    rng = random.Random(seed)
    output_chars = []
    for ch in text:
        if ch == "T" and rng.random() < flip_rate:
            output_chars.append("F")
        elif ch == "F" and rng.random() < flip_rate:
            output_chars.append("T")
        else:
            output_chars.append(ch)
    return "".join(output_chars)


def mask_question_blocks_in_response(line, block_indices, mask_char="X"):
    if not block_indices:
        return line

    if isinstance(block_indices, int):
        block_indices = {block_indices}
    else:
        block_indices = set(block_indices)

    blocks = line.split("||")
    masked_blocks = []
    for idx, block in enumerate(blocks, start=1):
        if idx in block_indices:
            masked_blocks.append("".join(mask_char if ch != " " else ch for ch in block))
        else:
            masked_blocks.append(block)
    return "||".join(masked_blocks)


def run_dumb_llm_job(
    codeword="chatgpt",
    variant="mini",
    system_prompt_file=r"llm_data/system_prompt.txt",
    exam_file=None,
    student_responses_file=r"llm_data/exam1/exam1_crawford.csv",
    batch_size=1,
    profile="very_low",
    max_output_tokens=None,
    reasoning_effort=None,
    thinking_budget_tokens=None,
    temperature=None,
    top_p=None,
    corruption_rate=None,
    mask_question_blocks=None,
    mask_char="X",
    output_file=None,
    seed=42,
):
    start_time = time.time()

    settings = merged_settings(
        profile=profile,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        thinking_budget_tokens=thinking_budget_tokens,
        temperature=temperature,
        top_p=top_p,
        corruption_rate=corruption_rate,
    )

    dispatch_table = {
        "chatgpt": call_chatgpt,
        "grok": call_grok,
        "deepseek": call_deepseek,
        "claude": call_claude,
        "gemini": call_gemini,
    }

    llm_function = dispatch_table.get(codeword.lower())
    if not llm_function:
        raise ValueError(f"Unknown codeword '{codeword}'.")

    with open(system_prompt_file, "r", encoding="utf-8") as f:
        base_system_prompt = f.read().strip()

    if exam_file:
        with open(exam_file, "r", encoding="utf-8") as f:
            exam_text = f.read().strip()
        base_system_prompt = (
            f"{base_system_prompt}\n\nEXAM (FIXED - DO NOT CHANGE):\n{exam_text}"
        )

    system_prompt = with_low_effort_overlay(base_system_prompt)

    student_text = csv_to_student_response_block(student_responses_file)
    if student_text.startswith("Error:"):
        raise ValueError(student_text)
    student_responses = student_text.splitlines()

    base_filename = os.path.splitext(os.path.basename(student_responses_file))[0]
    if output_file is None:
        masked_tag = ""
        if mask_question_blocks:
            if isinstance(mask_question_blocks, int):
                masked_tag = f"_masked_q{mask_question_blocks}"
            else:
                sorted_blocks = sorted(set(mask_question_blocks))
                masked_tag = "_masked_q" + "-".join(str(b) for b in sorted_blocks)

        output_file = (
            f"results_dumb_{codeword}_{variant}_{profile}_{batch_size}_{base_filename}{masked_tag}.txt"
        )

    print(f"--- DUMB LLM JOB ({codeword}/{variant}) ---")
    print(f"Students: {len(student_responses)} | Batch size: {batch_size}")
    print(
        "Settings: "
        f"effort={settings['reasoning_effort']}, "
        f"thinking_budget={settings['thinking_budget_tokens']}, "
        f"max_output={settings['max_output_tokens']}, "
        f"temp={settings['temperature']}, "
        f"top_p={settings['top_p']}, "
        f"flip_rate={settings['corruption_rate']}"
    )

    for i in range(0, len(student_responses), batch_size):
        batch_list = student_responses[i : i + batch_size]
        if mask_question_blocks:
            batch_list = [
                mask_question_blocks_in_response(
                    line,
                    block_indices=mask_question_blocks,
                    mask_char=mask_char,
                )
                for line in batch_list
            ]

        student_prompt = "Student Responses:\n" + "\n".join(batch_list)
        print(f"Processing Batch {i // batch_size + 1}...")

        with open(output_file, "a", encoding="utf-8") as f_out:
            print("-> Calling LLM")
            try:
                result = llm_function(
                    system_prompt=system_prompt,
                    student_prompt=student_prompt,
                    variant=variant,
                    settings=settings,
                )
                result = flip_tf_tokens(
                    text=result,
                    flip_rate=settings["corruption_rate"],
                    seed=seed + i,
                )
                f_out.write(f"{result}\n")
                print("OK: LLM returned")
            except Exception as exc:
                print(f"CRITICAL FAILURE on batch starting at index {i}: {exc}")

        time.sleep(1)

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print("\n--- JOB COMPLETE ---")
    print(f"Output file: {output_file}")
    print(f"Time spent running: {hours}h {minutes}m {seconds}s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run low-effort LLM experiments to emulate low-performing students."
    )
    parser.add_argument("--codeword", default="chatgpt", choices=["chatgpt", "grok", "deepseek", "claude", "gemini"])
    parser.add_argument("--variant", default="mini")
    parser.add_argument("--system-prompt-file", default=r"llm_data/system_prompt.txt")
    parser.add_argument("--exam-file")
    parser.add_argument("--student-responses-file", default=r"llm_data/exam1/exam1_crawford.csv")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--profile", default="very_low", choices=list(PROFILE_SETTINGS.keys()))
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"])
    parser.add_argument("--thinking-budget-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--corruption-rate", type=float)
    parser.add_argument("--mask-question-blocks", type=int, nargs="+")
    parser.add_argument("--mask-char", default="X")
    parser.add_argument("--output-file")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dumb_llm_job(
        codeword=args.codeword,
        variant=args.variant,
        system_prompt_file=args.system_prompt_file,
        exam_file=args.exam_file,
        student_responses_file=args.student_responses_file,
        batch_size=args.batch_size,
        profile=args.profile,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        thinking_budget_tokens=args.thinking_budget_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        corruption_rate=args.corruption_rate,
        mask_question_blocks=args.mask_question_blocks,
        mask_char=args.mask_char,
        output_file=args.output_file,
        seed=args.seed,
    )

