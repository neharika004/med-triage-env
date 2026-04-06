"""
FastAPI application exposing the MedicalTriageEnv via HTTP endpoints.
Compliant with OpenEnv spec — /reset accepts empty body OR JSON body.
"""

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from medical_triage_env import Action, MedicalTriageEnv

app = FastAPI(
    title="MedTriageEnv",
    description="OpenEnv-compliant Medical ED Triage Environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment store
_envs: Dict[str, MedicalTriageEnv] = {}

DEFAULT_TASK = "triage_easy"


def _get_env(task_name: str) -> MedicalTriageEnv:
    if task_name not in _envs:
        _envs[task_name] = MedicalTriageEnv(task_name=task_name)
    return _envs[task_name]


class StepRequest(BaseModel):
    task_name: str = DEFAULT_TASK
    action: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "MedTriageEnv",
        "description": "Medical Emergency Department Triage OpenEnv",
        "tasks": ["triage_easy", "triage_medium", "triage_hard"],
        "endpoints": ["/reset", "/step", "/state", "/grade", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset(request: Request):
    """
    Accept POST /reset with:
    - empty body
    - null body
    - JSON body: {} or {"task_name": "triage_easy"}
    """
    task_name = DEFAULT_TASK
    try:
        body_bytes = await request.body()
        if body_bytes and body_bytes.strip() not in (b"", b"null"):
            body = await request.json()
            if isinstance(body, dict):
                task_name = body.get("task_name", DEFAULT_TASK)
    except Exception:
        pass  # empty/null body — use default task

    env = _get_env(task_name)
    obs = env.reset()
    return {"observation": obs.dict(), "task": task_name}


@app.post("/step")
async def step(request: Request):
    """Accept step with task_name + action."""
    task_name = DEFAULT_TASK
    action_dict = {}
    try:
        body = await request.json()
        if isinstance(body, dict):
            task_name = body.get("task_name", DEFAULT_TASK)
            action_dict = body.get("action", {})
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON body: {e}")

    env = _get_env(task_name)
    try:
        action = Action(**action_dict)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(task_name: str = DEFAULT_TASK):
    env = _get_env(task_name)
    return env.state()


@app.get("/grade")
def grade(task_name: str = DEFAULT_TASK):
    env = _get_env(task_name)
    score = env.grade()
    return {"task": task_name, "score": score}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": "triage_easy",
                "description": "Single critical patient — assign urgency level",
                "difficulty": "easy",
                "max_steps": 5,
                "patients": 1,
            },
            {
                "name": "triage_medium",
                "description": "4-patient queue — triage + order tests + disposition",
                "difficulty": "medium",
                "max_steps": 15,
                "patients": 4,
            },
            {
                "name": "triage_hard",
                "description": "Full ED simulation — 6 patients, critical cases, full workflow",
                "difficulty": "hard",
                "max_steps": 40,
                "patients": 6,
            },
        ]
    }
