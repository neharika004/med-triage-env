# 🏥 MedTriageEnv — Medical ED Triage OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)](https://openenv.dev)
[![HuggingFace](https://img.shields.io/badge/🤗-Space-yellow)](https://huggingface.co/spaces)

## Overview

**MedTriageEnv** simulates an Emergency Department triage workflow — one of the most time-critical and high-stakes tasks in healthcare. An AI agent acts as an ED triage physician and must:

1. **Assess patients** using realistic vitals, chief complaints, and medical history
2. **Assign urgency** using the 5-level Emergency Severity Index (ESI)
3. **Order diagnostic tests** (ECG, troponin, CT, labs, etc.)
4. **Interpret results** as they become available
5. **Set disposition** (admit / discharge / observe / transfer)

### Why This Matters

Every year, over 140 million ED visits occur in the US alone. Triage errors — particularly **under-triaging critical patients** — contribute to preventable deaths. An AI agent trained on this environment could:
- Assist overburdened triage nurses with real-time decision support
- Flag high-risk patients who might otherwise be overlooked
- Standardize triage quality across institutions

---

## Tasks

| Task | Difficulty | Patients | Max Steps | Description |
|------|-----------|---------|-----------|-------------|
| `triage_easy` | Easy | 1 | 5 | Single STEMI patient — assign urgency |
| `triage_medium` | Medium | 4 | 15 | Multi-patient queue with varied acuity |
| `triage_hard` | Hard | 6 | 40 | Full ED: STEMI, SAH, meningitis, ectopic, CHF |

### Task Details

**triage_easy**
- One patient: 58M with crushing chest pain, hypotension, diaphoresis
- Agent must correctly classify ESI 1 (most urgent)
- Tests: ECG, troponin, chest X-ray, labs
- Expected score for a competent agent: 0.8–1.0

**triage_medium**
- Four patients with varying acuity (ESI 1, 2, 4, 5)
- Agent must triage all, order appropriate tests, set dispositions
- Partial credit for correct urgency and test selection
- Expected score: 0.6–0.85

**triage_hard**
- Six patients including life-threatening emergencies:
  - STEMI (ESI 1), Subarachnoid Hemorrhage (ESI 1), Bacterial Meningitis in a child (ESI 1)
  - Ectopic Pregnancy with hemodynamic instability (ESI 1)
  - Decompensated CHF (ESI 2), Uncomplicated UTI (ESI 5)
- Must correctly prioritize, test, and disposition all patients
- Under-triaging ESI-1 patients applies severe penalties
- Expected score for frontier models: 0.5–0.75

---

## Observation Space

```python
class Observation(BaseModel):
    task_name: str                              # Current task
    step: int                                   # Current step number
    patients: List[Patient]                     # Full patient data
    ordered_tests: Dict[str, List[str]]         # Tests ordered per patient
    test_results: Dict[str, Dict[str, str]]     # Results as they return
    assigned_urgency: Dict[str, int]            # ESI assignments made
    dispositions: Dict[str, str]                # Dispositions set
    elapsed_minutes: int                        # Episode time
    message: str                                # Feedback from last action
    available_actions: List[str]                # What agent can do next

class Patient(BaseModel):
    patient_id: str
    age: int
    gender: str
    chief_complaint: str
    vitals: VitalSigns                          # HR, BP, SpO2, RR, Temp, Pain
    history: str                                # PMH, social history
    allergies: List[str]
    medications: List[str]
    arrival_time: int                           # Minutes since episode start
```

---

## Action Space

```python
class Action(BaseModel):
    action_type: str        # assign_urgency | order_test | set_disposition | reassess | done
    patient_id: str         # Required for patient-specific actions
    urgency_level: int      # 1-5 (for assign_urgency)
    test_name: str          # (for order_test) — see valid tests below
    disposition: str        # admit | discharge | observe | transfer
```

**Valid Tests:** ECG, troponin, chest_xray, CBC, BMP, CT_head, lumbar_puncture, coagulation, ankle_xray, blood_culture, beta_hCG, pelvic_ultrasound, blood_type, BNP, urinalysis, urine_culture, rapid_strep

---

## Reward Function

The reward is **shaped throughout the episode** (not sparse):

| Signal | Reward | When |
|--------|--------|------|
| Correct urgency assignment | +0.30 × accuracy | Immediate |
| Ordering an indicated test | +0.15 | Immediate |
| Duplicate test ordered | −0.05 | Immediate |
| Unknown test | −0.05 | Immediate |
| Correct disposition | +0.30 | Immediate |
| Wrong disposition (sick patient discharged) | −0.20 | Immediate |
| End-of-episode: urgency accuracy | 0–0.30 | Final |
| End-of-episode: test appropriateness | 0–0.35 | Final |
| End-of-episode: disposition accuracy | 0–0.25 | Final |
| Critical patient safety bonus | ±0.10 | Final |

**Final score formula:**
```
score = 0.30 × urgency_accuracy
      + 0.35 × test_appropriateness  
      + 0.25 × disposition_accuracy
      + clamp(safety_bonus, -0.30, 0.10)
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| GET | `/` | Environment info |
| GET | `/health` | Health check |
| POST | `/reset` | Start new episode |
| POST | `/step` | Take one action |
| GET | `/state` | Current state |
| GET | `/grade` | Current score |
| GET | `/tasks` | List tasks |

### Example Usage

```python
import requests

BASE = "https://your-space.hf.space"

# Start episode
obs = requests.post(f"{BASE}/reset", json={"task_name": "triage_easy"}).json()

# Take action
result = requests.post(f"{BASE}/step", json={
    "task_name": "triage_easy",
    "action": {
        "action_type": "assign_urgency",
        "patient_id": "P001",
        "urgency_level": 1
    }
}).json()

print(result["reward"])  # {"value": 0.3, "breakdown": {...}, "message": "..."}
```

---

## Setup & Installation

### Local

```bash
git clone https://github.com/your-username/med-triage-env
cd med-triage-env
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t med-triage-env .
docker run -p 7860:7860 med-triage-env
```

### Run Inference

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

python inference.py
```

---

## Baseline Scores

Tested with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router:

| Task | Score | Notes |
|------|-------|-------|
| triage_easy | 0.72 | Correctly identifies STEMI urgency, orders ECG+troponin |
| triage_medium | 0.58 | Struggles with test selection for all 4 patients |
| triage_hard | 0.41 | Misses some critical safety priorities under time pressure |

---

## ESI Reference

| Level | Description | Example |
|-------|-------------|---------|
| ESI 1 | Immediate, life-saving intervention required | Cardiac arrest, STEMI, status epilepticus |
| ESI 2 | High risk situation, severe pain/distress | Chest pain, altered mental status |
| ESI 3 | Stable, needs 2+ resources | Abdominal pain, complex lacerations |
| ESI 4 | Stable, needs 1 resource | Ankle sprain, simple UTI |
| ESI 5 | Stable, no resources needed | Minor cold, prescription refill |

---

## License

MIT License — see LICENSE file.
