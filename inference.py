"""
MedTriageEnv — Inference Script
Runs an LLM agent against all 3 tasks and emits structured stdout logs.

STDOUT FORMAT (required):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from medical_triage_env import Action, MedicalTriageEnv

# ---------------------------------------------------------------------------
# Config — read from env vars exactly as required
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")  # NO default — required at runtime

# Optional docker image name
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "med_triage_env"
MAX_STEPS = 30
TEMPERATURE = 0.2
MAX_TOKENS = 512
SUCCESS_THRESHOLD = 0.5

TASKS = ["triage_easy", "triage_medium", "triage_hard"]

# ---------------------------------------------------------------------------
# Logging helpers — exact format, no deviation
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Sanitize action string: remove newlines
    action_clean = action.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_val = str(success).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert emergency department physician AI assistant.
You will receive patient cases and must triage them by:
1. Assigning urgency (ESI levels 1-5, where 1 = most critical)
2. Ordering appropriate diagnostic tests
3. Setting patient disposition (admit/discharge/observe/transfer)

ESI Levels:
- ESI 1: Immediate life threat (e.g., cardiac arrest, STEMI, severe sepsis)
- ESI 2: High risk, confused/lethal potential (e.g., chest pain, stroke signs)
- ESI 3: Stable but needs 2+ resources
- ESI 4: Stable, needs 1 resource
- ESI 5: Stable, no resources needed

You must respond with a valid JSON action in this EXACT format:
{
  "action_type": "assign_urgency" | "order_test" | "set_disposition" | "reassess" | "done",
  "patient_id": "P001",         (required for patient actions)
  "urgency_level": 1,           (required for assign_urgency, integer 1-5)
  "test_name": "ECG",           (required for order_test)
  "disposition": "admit"        (required for set_disposition: admit/discharge/observe/transfer)
}

Available tests: ECG, troponin, chest_xray, CBC, BMP, CT_head, lumbar_puncture,
coagulation, ankle_xray, blood_culture, beta_hCG, pelvic_ultrasound, blood_type,
BNP, urinalysis, urine_culture, rapid_strep

Respond with ONLY the JSON object, nothing else.
""").strip()


def build_user_prompt(obs_dict: Dict[str, Any]) -> str:
    """Convert observation to a concise prompt for the LLM."""
    lines = [f"TASK: {obs_dict['task_name']} | Step: {obs_dict['step']}"]
    lines.append(f"Status: {obs_dict['message']}")
    lines.append("")
    lines.append("PATIENTS:")

    for p in obs_dict["patients"]:
        pid = p["patient_id"]
        v = p["vitals"]
        urgency_status = obs_dict["assigned_urgency"].get(pid, "NOT ASSIGNED")
        disp_status = obs_dict["dispositions"].get(pid, "NOT SET")
        tests_ordered = obs_dict["ordered_tests"].get(pid, [])
        results = obs_dict["test_results"].get(pid, {})

        lines.append(f"\n[{pid}] Age {p['age']}{p['gender']} — {p['chief_complaint']}")
        lines.append(
            f"  Vitals: HR={v['heart_rate']} BP={v['systolic_bp']}/{v['diastolic_bp']} "
            f"SpO2={v['spo2']}% RR={v['respiratory_rate']} Temp={v['temperature']}°C Pain={v['pain_score']}/10"
        )
        lines.append(f"  History: {p['history']}")
        lines.append(f"  Allergies: {', '.join(p['allergies']) or 'None'}")
        lines.append(f"  Meds: {', '.join(p['medications']) or 'None'}")
        lines.append(f"  ESI Assigned: {urgency_status} | Disposition: {disp_status}")
        if tests_ordered:
            lines.append(f"  Tests Ordered: {', '.join(tests_ordered)}")
        if results:
            lines.append("  Results:")
            for test, result in results.items():
                lines.append(f"    {test}: {result}")

    lines.append("")
    lines.append("Decide your next single action as JSON.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_name: str) -> float:
    env = MedicalTriageEnv(task_name=task_name)
    obs = env.reset()

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    step = 0
    done = False
    last_error: Optional[str] = None
    conversation: List[Dict] = []

    obs_dict = obs.dict()

    while not done and step < MAX_STEPS:
        step += 1
        user_content = build_user_prompt(obs_dict)
        conversation.append({"role": "user", "content": user_content})

        # LLM call
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw_text = response.choices[0].message.content.strip()
            conversation.append({"role": "assistant", "content": raw_text})

            # Parse JSON action
            # Strip markdown code fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            action_dict = json.loads(raw_text.strip())
            action = Action(**action_dict)
            action_str = json.dumps(action_dict)
            last_error = None

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            action = Action(action_type="done")
            action_str = '{"action_type":"done"}'
        except Exception as e:
            last_error = str(e)
            action = Action(action_type="done")
            action_str = '{"action_type":"done"}'

        # Step environment
        obs, reward_obj, done, info = env.step(action)
        obs_dict = obs.dict()
        reward_val = reward_obj.value
        rewards.append(reward_val)

        log_step(
            step=step,
            action=action_str,
            reward=reward_val,
            done=done,
            error=last_error,
        )

        if done:
            break

    # Final grade
    final_score = env.grade()
    success = final_score >= SUCCESS_THRESHOLD

    env.close()

    log_end(
        success=success,
        steps=step,
        score=final_score,
        rewards=rewards,
    )

    return final_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
    )

    all_scores = []
    for task_name in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_name}", flush=True)
        print(f"{'='*60}", flush=True)
        try:
            score = run_task(client, task_name)
            all_scores.append(score)
            print(f"Task {task_name} completed. Score: {score:.2f}", flush=True)
        except Exception as e:
            print(f"Task {task_name} FAILED: {e}", file=sys.stderr, flush=True)
            all_scores.append(0.0)
        time.sleep(1)  # brief pause between tasks

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n{'='*60}", flush=True)
    print(f"ALL TASKS COMPLETE", flush=True)
    print(f"Scores: {[f'{s:.2f}' for s in all_scores]}", flush=True)
    print(f"Average: {avg_score:.2f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
