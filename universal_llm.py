import os
import time
from openai import OpenAI
from google import genai
from google.genai import types
import anthropic
from tenacity import retry, stop_after_attempt, wait_random_exponential, before_sleep_log
from extract_exam_scores import csv_to_student_response_block
import logging
import sys

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# This decorator defines the backoff strategy:
# 1. Wait: Random exponential backoff (min 1s, max 60s)
# 2. Stop: Give up after 6 attempts
# 3. Trigger: Retries on any generic Exception (covers network, 429, 503)
#    (For tighter control, you can list specific exceptions like RateLimitError)
robust_retry = retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    reraise=True,  # If it fails 6 times, raise the final error
    before_sleep=before_sleep_log(logger, logging.INFO) # <--- THIS PRINTS RETRIES
)

# ---------- OpenAI / ChatGPT ----------
@robust_retry
def call_chatgpt(system_prompt, student_prompt, variant = "mini"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if variant == "mini":
        model_name = f"gpt-5-mini"
    else:
        model_name = f"gpt-5.2"

    response = client.responses.create(
        model=model_name,
        reasoning={"effort": "high"},
        prompt_cache_key="first_cache",
        max_output_tokens=25000,
        instructions=system_prompt,
        input=[
            {
                "role": "user", 
                "content": student_prompt
            }
        ]
    )
    return response.output_text.strip()


# ---------- Grok (xAI) ----------
@robust_retry
def call_grok(system_prompt, student_prompt, variant="4"):
    from xai_sdk import Client
    from xai_sdk.chat import user, system

    client = Client( 
        api_key=os.getenv("XAI_API_KEY")
    ) 
    model_name = "grok-4-1-fast-reasoning" if variant == "fast-reasoning" else "grok-4-latest"

    chat = client.chat.create(
        model=model_name,
        store_messages=False,
        temperature=0
    )

    chat.append(system(system_prompt))
    chat.append(user(student_prompt))
    response = chat.sample()

    return response.content.strip()


# ---------- DeepSeek via OpenRouter ----------
@robust_retry
def call_deepseek(system_prompt, student_prompt, variant = "v3.2"):
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    model_name = f"deepseek/deepseek-{variant}"

    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        max_tokens=13000,
        top_p=1.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": student_prompt},
        ],
         extra_body={
            "reasoning": {
                "effort": "high",
                "exclude": True
            }
        }
    )
    return response.choices[0].message.content.strip()


# ---------- Claude ----------
@robust_retry
def call_claude(system_prompt, student_prompt, variant = "haiku"):
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model_name = f"claude-{variant}-4-5"

    response = client.messages.create(
        model=model_name,
        max_tokens=13000,
        top_p=1.0,
        system=system_prompt,
        messages=[
            {"role": "user", "content": student_prompt}
        ],
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
        },
    )
    for block in response.content:
        if block.type == "text":
            return block.text.strip()
    return ""


# ---------- Gemini ----------
@robust_retry
def call_gemini(system_prompt, student_prompt, variant="flash"):
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
        http_options={'timeout': 120000}
    )
    model_name = "gemini-3-flash-preview" if variant == "flash" else "gemini-3-pro-preview"
    
    response = client.models.generate_content(
        model=model_name,
        contents=[student_prompt],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt, 
            thinking_config=types.ThinkingConfig(
                includeThoughts=False,  # Set True if you want to see the "scratchpad"
            )
        )
    )
    return response.text.strip()
import time


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
            # Replace any non-space character with mask_char, preserve spacing
            masked_blocks.append("".join(mask_char if ch != " " else ch for ch in block))
        else:
            masked_blocks.append(block)
    return "||".join(masked_blocks)


def run_llm_job(
    codeword="gemini",
    variant="pro",
    system_prompt_file=r"llm_data/system_prompt.txt",
    exam_file=None,
    student_responses_file=r"llm_data/exam1/exam1_ghosh.csv",
    batch_size=3,
    mask_question_blocks=None,
    mask_char="X",
    output_file=None,
):
    """
    Runs LLM calls with a fixed SYSTEM prompt and per-line STUDENT responses.
    """

    start_time = time.time()

    dispatch_table = {
        "chatgpt": call_chatgpt,
        "grok": call_grok,
        "deepseek": call_deepseek,
        "claude": call_claude,
        "gemini": call_gemini,
    }

    llm_function = dispatch_table.get(codeword.lower())
    if not llm_function:
        raise ValueError(f"Unknown codeword '{codeword}'")

    # Load system prompt (ONCE)
    with open(system_prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    # Optionally append fixed exam context to the system instructions
    if exam_file:
        with open(exam_file, "r", encoding="utf-8") as f:
            exam_text = f.read().strip()
        system_prompt = f"{system_prompt}\n\nEXAM (FIXED - DO NOT CHANGE):\n{exam_text}"

    # Load student responses
    student_responses = csv_to_student_response_block(student_responses_file).splitlines()

    print(f"--- {codeword.upper()} JOB ---")
    print(f"Students: {len(student_responses)} | Batch size: {batch_size}")

    base_filename = os.path.splitext(os.path.basename(student_responses_file))[0]
    if output_file is None:
        masked_tag = ""
        if mask_question_blocks:
            if isinstance(mask_question_blocks, int):
                masked_tag = f"_masked_q{mask_question_blocks}"
            else:
                sorted_blocks = sorted(set(mask_question_blocks))
                masked_tag = "_masked_q" + "-".join(str(b) for b in sorted_blocks)
        output_file = f"results_{codeword}_{variant}_{batch_size}_{base_filename}{masked_tag}.txt"
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
        
        # 2. Join them into one single string
        student_prompt = "Student Responses:\n" + "\n".join(batch_list)

        print(f"Processing Batch {i // batch_size + 1}...")

        with open(output_file, "a", encoding="utf-8") as f_out:
            # The retry logic happens INSIDE this function call 
            # If it fails, it will hang here while retrying, then proceed or crash after 6 tries.
            print("-> Calling LLM")
            try:
                result = llm_function(system_prompt, student_prompt, variant=variant)
                f_out.write(f"{result}\n")
            except Exception as e:
                print(f"CRITICAL FAILURE on batch starting at index {i}: {e}")
                # Optional: break or continue depending on preference
            print("OK: LLM returned")
        # Polite pacing
        time.sleep(1)

    print("\n--- JOB COMPLETE ---")
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"Time spent running: {hours}h {minutes}m {seconds}s")

if __name__ == "__main__":
    start_time = time.time()
    run_llm_job(
        codeword="chatgpt",
        variant="5.2",
        system_prompt_file=r"llm_data/system_prompt.txt",
        exam_file=None,
        student_responses_file=r"llm_data/exam1/exam1_crawford.csv",
        batch_size=1,
    )
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"Time spent running: {hours}h {minutes}m {seconds}s")
