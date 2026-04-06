"""
MedTriageEnv — OpenEnv-compliant Medical Triage Environment

Real-world task: Emergency department nurses and doctors must rapidly assess
incoming patients, assign urgency levels, order appropriate tests, and
decide on disposition. This environment simulates that decision pipeline.

Three tasks:
  1. triage_easy   — single patient, classify urgency (1-5 scale)
  2. triage_medium — multi-patient queue, prioritize + order tests
  3. triage_hard   — full ED scenario: triage + tests + disposition + follow-up
"""

from __future__ import annotations

import copy
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Typed Models
# ---------------------------------------------------------------------------

class VitalSigns(BaseModel):
    heart_rate: int          # bpm
    systolic_bp: int         # mmHg
    diastolic_bp: int        # mmHg
    spo2: float              # %
    respiratory_rate: int    # breaths/min
    temperature: float       # Celsius
    pain_score: int          # 0-10

class Patient(BaseModel):
    patient_id: str
    age: int
    gender: str
    chief_complaint: str
    vitals: VitalSigns
    history: str
    allergies: List[str]
    medications: List[str]
    arrival_time: int        # minutes since episode start
    true_urgency: int        # ground truth ESI level 1-5 (1=most urgent)
    true_diagnosis: str
    required_tests: List[str]
    correct_disposition: str  # admit / discharge / observe / transfer

class Observation(BaseModel):
    task_name: str
    step: int
    patients: List[Patient]
    ordered_tests: Dict[str, List[str]]      # patient_id -> tests ordered
    test_results: Dict[str, Dict[str, str]]  # patient_id -> test -> result
    assigned_urgency: Dict[str, int]         # patient_id -> ESI level assigned
    dispositions: Dict[str, str]             # patient_id -> disposition
    elapsed_minutes: int
    message: str
    available_actions: List[str]

class Action(BaseModel):
    action_type: str   # assign_urgency | order_test | set_disposition | reassess | done
    patient_id: Optional[str] = None
    urgency_level: Optional[int] = None     # 1-5
    test_name: Optional[str] = None
    disposition: Optional[str] = None       # admit | discharge | observe | transfer

class Reward(BaseModel):
    value: float
    breakdown: Dict[str, float]
    message: str

# ---------------------------------------------------------------------------
# Patient Case Bank
# ---------------------------------------------------------------------------

CASE_BANK: List[Dict] = [
    {
        "patient_id": "P001",
        "age": 58,
        "gender": "M",
        "chief_complaint": "Crushing chest pain radiating to left arm, started 30 min ago",
        "vitals": {"heart_rate": 110, "systolic_bp": 88, "diastolic_bp": 60,
                   "spo2": 94.0, "respiratory_rate": 22, "temperature": 37.1, "pain_score": 9},
        "history": "Known hypertension, smoker 30 pack-years, hyperlipidemia",
        "allergies": ["penicillin"],
        "medications": ["lisinopril", "atorvastatin"],
        "arrival_time": 0,
        "true_urgency": 1,
        "true_diagnosis": "STEMI",
        "required_tests": ["ECG", "troponin", "chest_xray", "CBC", "BMP"],
        "correct_disposition": "admit"
    },
    {
        "patient_id": "P002",
        "age": 24,
        "gender": "F",
        "chief_complaint": "Mild sore throat and low-grade fever for 2 days",
        "vitals": {"heart_rate": 82, "systolic_bp": 118, "diastolic_bp": 76,
                   "spo2": 99.0, "respiratory_rate": 16, "temperature": 37.9, "pain_score": 3},
        "history": "No significant PMH",
        "allergies": [],
        "medications": [],
        "arrival_time": 5,
        "true_urgency": 5,
        "true_diagnosis": "Viral pharyngitis",
        "required_tests": ["rapid_strep"],
        "correct_disposition": "discharge"
    },
    {
        "patient_id": "P003",
        "age": 72,
        "gender": "F",
        "chief_complaint": "Sudden onset severe headache, worst of my life, neck stiffness",
        "vitals": {"heart_rate": 96, "systolic_bp": 172, "diastolic_bp": 98,
                   "spo2": 97.0, "respiratory_rate": 18, "temperature": 38.2, "pain_score": 10},
        "history": "Hypertension, no prior headache history",
        "allergies": ["sulfa"],
        "medications": ["amlodipine"],
        "arrival_time": 2,
        "true_urgency": 1,
        "true_diagnosis": "Subarachnoid hemorrhage",
        "required_tests": ["CT_head", "lumbar_puncture", "CBC", "BMP", "coagulation"],
        "correct_disposition": "admit"
    },
    {
        "patient_id": "P004",
        "age": 45,
        "gender": "M",
        "chief_complaint": "Right ankle swelling and pain after twisting it playing basketball",
        "vitals": {"heart_rate": 78, "systolic_bp": 128, "diastolic_bp": 82,
                   "spo2": 99.0, "respiratory_rate": 14, "temperature": 36.8, "pain_score": 5},
        "history": "No significant PMH",
        "allergies": [],
        "medications": ["ibuprofen PRN"],
        "arrival_time": 10,
        "true_urgency": 4,
        "true_diagnosis": "Ankle sprain, r/o fracture",
        "required_tests": ["ankle_xray"],
        "correct_disposition": "discharge"
    },
    {
        "patient_id": "P005",
        "age": 8,
        "gender": "M",
        "chief_complaint": "High fever, stiff neck, photophobia, rash",
        "vitals": {"heart_rate": 132, "systolic_bp": 92, "diastolic_bp": 58,
                   "spo2": 96.0, "respiratory_rate": 28, "temperature": 39.8, "pain_score": 8},
        "history": "Previously healthy, unvaccinated",
        "allergies": [],
        "medications": [],
        "arrival_time": 1,
        "true_urgency": 1,
        "true_diagnosis": "Bacterial meningitis",
        "required_tests": ["CBC", "BMP", "blood_culture", "lumbar_puncture", "CT_head"],
        "correct_disposition": "admit"
    },
    {
        "patient_id": "P006",
        "age": 33,
        "gender": "F",
        "chief_complaint": "Severe abdominal pain, missed period, positive home pregnancy test",
        "vitals": {"heart_rate": 118, "systolic_bp": 94, "diastolic_bp": 62,
                   "spo2": 98.0, "respiratory_rate": 20, "temperature": 37.0, "pain_score": 9},
        "history": "G1P0, LMP 7 weeks ago",
        "allergies": [],
        "medications": ["prenatal vitamins"],
        "arrival_time": 3,
        "true_urgency": 1,
        "true_diagnosis": "Ectopic pregnancy",
        "required_tests": ["beta_hCG", "pelvic_ultrasound", "CBC", "BMP", "blood_type"],
        "correct_disposition": "admit"
    },
    {
        "patient_id": "P007",
        "age": 67,
        "gender": "M",
        "chief_complaint": "Shortness of breath, worsening over 3 days, bilateral leg swelling",
        "vitals": {"heart_rate": 104, "systolic_bp": 148, "diastolic_bp": 92,
                   "spo2": 91.0, "respiratory_rate": 24, "temperature": 37.3, "pain_score": 4},
        "history": "CHF, CKD stage 3, DM type 2",
        "allergies": ["ACE inhibitors"],
        "medications": ["furosemide", "carvedilol", "spironolactone", "metformin"],
        "arrival_time": 8,
        "true_urgency": 2,
        "true_diagnosis": "Acute decompensated heart failure",
        "required_tests": ["BNP", "chest_xray", "ECG", "CBC", "BMP", "troponin"],
        "correct_disposition": "admit"
    },
    {
        "patient_id": "P008",
        "age": 19,
        "gender": "F",
        "chief_complaint": "Burning urination, frequency, mild pelvic discomfort for 1 day",
        "vitals": {"heart_rate": 76, "systolic_bp": 114, "diastolic_bp": 70,
                   "spo2": 99.0, "respiratory_rate": 14, "temperature": 37.2, "pain_score": 3},
        "history": "Sexually active, no prior UTIs",
        "allergies": [],
        "medications": ["OCP"],
        "arrival_time": 20,
        "true_urgency": 5,
        "true_diagnosis": "Uncomplicated UTI",
        "required_tests": ["urinalysis", "urine_culture"],
        "correct_disposition": "discharge"
    },
]

TEST_RESULTS_BANK: Dict[str, Dict[str, str]] = {
    "P001": {
        "ECG": "ST elevation in leads II, III, aVF, V4-V6 — STEMI pattern",
        "troponin": "Troponin I: 2.8 ng/mL (elevated, ref <0.04)",
        "chest_xray": "Mild pulmonary vascular congestion",
        "CBC": "WBC 12.4, Hgb 14.2, Plt 224",
        "BMP": "Na 138, K 4.1, Cr 1.0, BUN 18, Glucose 142",
    },
    "P002": {
        "rapid_strep": "Negative — likely viral etiology",
    },
    "P003": {
        "CT_head": "Hyperdense blood in basal cisterns — subarachnoid hemorrhage",
        "lumbar_puncture": "Xanthochromia present, RBC 85,000",
        "CBC": "WBC 14.2, Hgb 13.8, Plt 198",
        "BMP": "Na 136, K 3.8, Cr 0.9",
        "coagulation": "PT/INR 1.1, PTT 28",
    },
    "P004": {
        "ankle_xray": "No acute fracture — soft tissue swelling consistent with sprain",
    },
    "P005": {
        "CBC": "WBC 22.4 (elevated, left shift), Hgb 11.8, Plt 156",
        "BMP": "Na 134, K 3.6, Cr 0.5, Glucose 98",
        "blood_culture": "Pending — gram stain shows gram-negative diplococci",
        "lumbar_puncture": "CSF: WBC 850, protein 180, glucose 28 — bacterial meningitis pattern",
        "CT_head": "No mass lesion, no herniation — safe for LP",
    },
    "P006": {
        "beta_hCG": "Beta hCG: 6,200 mIU/mL",
        "pelvic_ultrasound": "No intrauterine pregnancy, free fluid in pelvis, adnexal mass 3.2cm",
        "CBC": "WBC 11.2, Hgb 10.4 (low), Plt 312",
        "BMP": "Na 137, K 3.9, Cr 0.7",
        "blood_type": "O negative — administer Rh immunoglobulin",
    },
    "P007": {
        "BNP": "BNP: 1,840 pg/mL (severely elevated, ref <100)",
        "chest_xray": "Bilateral pulmonary edema, cardiomegaly, pleural effusions",
        "ECG": "Sinus tachycardia, LVH pattern, no acute ischemia",
        "CBC": "WBC 9.2, Hgb 11.8, Plt 204",
        "BMP": "Na 132, K 5.2, Cr 2.1 (elevated from baseline 1.6), BUN 42",
        "troponin": "Troponin I: 0.08 (mildly elevated, demand ischemia)",
    },
    "P008": {
        "urinalysis": "UA: WBC 50+, nitrites positive, leukocyte esterase 3+",
        "urine_culture": "Pending — empiric treatment appropriate",
    },
}

VALID_TESTS = [
    "ECG", "troponin", "chest_xray", "CBC", "BMP", "CT_head", "lumbar_puncture",
    "coagulation", "ankle_xray", "blood_culture", "beta_hCG", "pelvic_ultrasound",
    "blood_type", "BNP", "urinalysis", "urine_culture", "rapid_strep"
]
VALID_DISPOSITIONS = ["admit", "discharge", "observe", "transfer"]

# ---------------------------------------------------------------------------
# Grader Helpers
# ---------------------------------------------------------------------------

def _urgency_score(assigned: int, true: int) -> float:
    """Score urgency assignment. ESI 1 misses are most penalized."""
    diff = abs(assigned - true)
    if diff == 0:
        return 1.0
    if true == 1 and diff >= 2:
        return 0.0          # dangerous under-triage of critical patient
    if diff == 1:
        return 0.5
    if diff == 2:
        return 0.2
    return 0.0

def _test_score(ordered: List[str], required: List[str]) -> float:
    """Partial credit for ordering the right tests."""
    if not required:
        return 1.0
    ordered_set = set(ordered)
    required_set = set(required)
    correct = len(ordered_set & required_set)
    extra = max(0, len(ordered_set) - len(required_set))
    recall = correct / len(required_set)
    penalty = min(0.3, extra * 0.05)   # small penalty for over-ordering
    return max(0.0, recall - penalty)

def _disposition_score(assigned: str, correct: str) -> float:
    return 1.0 if assigned == correct else 0.0

# ---------------------------------------------------------------------------
# Task Definitions
# ---------------------------------------------------------------------------

class Task:
    def __init__(self, name: str, patient_ids: List[str], max_steps: int):
        self.name = name
        self.patient_ids = patient_ids
        self.max_steps = max_steps

TASKS: Dict[str, Task] = {
    "triage_easy": Task(
        name="triage_easy",
        patient_ids=["P001"],
        max_steps=5
    ),
    "triage_medium": Task(
        name="triage_medium",
        patient_ids=["P001", "P003", "P004", "P008"],
        max_steps=15
    ),
    "triage_hard": Task(
        name="triage_hard",
        patient_ids=["P001", "P003", "P005", "P006", "P007", "P008"],
        max_steps=40
    ),
}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MedicalTriageEnv:
    """
    OpenEnv-compliant Medical Emergency Department Triage Environment.

    The agent acts as an ED triage system that must:
      1. Assign urgency levels (ESI 1-5) to incoming patients
      2. Order appropriate diagnostic tests
      3. Interpret results and set patient disposition
    """

    def __init__(self, task_name: str = "triage_easy"):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}. Choose from {list(TASKS.keys())}")
        self.task = TASKS[task_name]
        self._step_count = 0
        self._done = False
        self._patients: Dict[str, Patient] = {}
        self._ordered_tests: Dict[str, List[str]] = {}
        self._test_results: Dict[str, Dict[str, str]] = {}
        self._assigned_urgency: Dict[str, int] = {}
        self._dispositions: Dict[str, str] = {}
        self._cumulative_reward = 0.0
        self._episode_start = time.time()

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        self._step_count = 0
        self._done = False
        self._ordered_tests = {}
        self._test_results = {}
        self._assigned_urgency = {}
        self._dispositions = {}
        self._cumulative_reward = 0.0
        self._episode_start = time.time()

        # Load patients for this task
        self._patients = {}
        for pid in self.task.patient_ids:
            case = next(c for c in CASE_BANK if c["patient_id"] == pid)
            self._patients[pid] = Patient(
                patient_id=case["patient_id"],
                age=case["age"],
                gender=case["gender"],
                chief_complaint=case["chief_complaint"],
                vitals=VitalSigns(**case["vitals"]),
                history=case["history"],
                allergies=case["allergies"],
                medications=case["medications"],
                arrival_time=case["arrival_time"],
                true_urgency=case["true_urgency"],
                true_diagnosis=case["true_diagnosis"],
                required_tests=case["required_tests"],
                correct_disposition=case["correct_disposition"],
            )
            self._ordered_tests[pid] = []
            self._test_results[pid] = {}

        return self._build_observation("Episode started. Assess all patients.")

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self._done:
            obs = self._build_observation("Episode already finished.")
            reward = Reward(value=0.0, breakdown={}, message="Episode done.")
            return obs, reward, True, {}

        self._step_count += 1
        reward_val = 0.0
        breakdown: Dict[str, float] = {}
        message = ""

        # --- Action dispatch ---
        if action.action_type == "assign_urgency":
            r, msg = self._handle_assign_urgency(action)
            reward_val = r
            breakdown["urgency"] = r
            message = msg

        elif action.action_type == "order_test":
            r, msg = self._handle_order_test(action)
            reward_val = r
            breakdown["test_order"] = r
            message = msg

        elif action.action_type == "set_disposition":
            r, msg = self._handle_disposition(action)
            reward_val = r
            breakdown["disposition"] = r
            message = msg

        elif action.action_type == "reassess":
            r, msg = self._handle_reassess(action)
            reward_val = r
            breakdown["reassess"] = r
            message = msg

        elif action.action_type == "done":
            reward_val, breakdown, message = self._compute_final_reward()
            self._done = True

        else:
            reward_val = -0.1
            breakdown["invalid"] = -0.1
            message = f"Unknown action type: {action.action_type}. Penalty applied."

        # Step limit
        if self._step_count >= self.task.max_steps and not self._done:
            final_r, final_b, final_msg = self._compute_final_reward()
            reward_val += final_r * 0.5  # partial credit for hitting limit
            breakdown.update(final_b)
            message += f" | Step limit reached: {final_msg}"
            self._done = True

        self._cumulative_reward += reward_val
        reward_obj = Reward(
            value=round(reward_val, 4),
            breakdown=breakdown,
            message=message
        )
        obs = self._build_observation(message)
        return obs, reward_obj, self._done, {"step": self._step_count}

    def state(self) -> Dict[str, Any]:
        return {
            "task": self.task.name,
            "step": self._step_count,
            "done": self._done,
            "patients": {pid: p.dict() for pid, p in self._patients.items()},
            "ordered_tests": self._ordered_tests,
            "test_results": self._test_results,
            "assigned_urgency": self._assigned_urgency,
            "dispositions": self._dispositions,
            "cumulative_reward": round(self._cumulative_reward, 4),
        }

    def close(self) -> None:
        self._done = True

    # ------------------------------------------------------------------
    # Action Handlers
    # ------------------------------------------------------------------

    def _handle_assign_urgency(self, action: Action) -> Tuple[float, str]:
        pid = action.patient_id
        level = action.urgency_level
        if pid not in self._patients:
            return -0.1, f"Patient {pid} not found."
        if level not in range(1, 6):
            return -0.1, f"Invalid urgency level {level}. Must be 1-5."
        if pid in self._assigned_urgency:
            # Allow reassignment with small penalty
            old = self._assigned_urgency[pid]
            self._assigned_urgency[pid] = level
            true = self._patients[pid].true_urgency
            score = _urgency_score(level, true) * 0.5  # partial for re-assign
            return score, f"Patient {pid}: urgency reassigned {old}→{level}. Partial credit."
        self._assigned_urgency[pid] = level
        true = self._patients[pid].true_urgency
        score = _urgency_score(level, true)
        if score == 1.0:
            return score * 0.3, f"Patient {pid}: urgency {level} — CORRECT (ESI {true})."
        elif score > 0:
            return score * 0.3, f"Patient {pid}: urgency {level} — partial (true ESI {true})."
        else:
            msg = f"Patient {pid}: urgency {level} — WRONG (true ESI {true})."
            if self._patients[pid].true_urgency == 1:
                msg += " CRITICAL PATIENT UNDER-TRIAGED!"
            return 0.0, msg

    def _handle_order_test(self, action: Action) -> Tuple[float, str]:
        pid = action.patient_id
        test = action.test_name
        if pid not in self._patients:
            return -0.1, f"Patient {pid} not found."
        if test not in VALID_TESTS:
            return -0.05, f"Unknown test '{test}'. No result."
        if test in self._ordered_tests[pid]:
            return -0.05, f"Test '{test}' already ordered for {pid}. Duplicate — small penalty."
        self._ordered_tests[pid].append(test)
        # Reveal result if available
        result_text = ""
        if pid in TEST_RESULTS_BANK and test in TEST_RESULTS_BANK[pid]:
            self._test_results[pid][test] = TEST_RESULTS_BANK[pid][test]
            result_text = f" Result: {TEST_RESULTS_BANK[pid][test]}"
        required = self._patients[pid].required_tests
        if test in required:
            return 0.15, f"Test '{test}' ordered for {pid} — appropriate.{result_text}"
        else:
            return -0.02, f"Test '{test}' ordered for {pid} — not indicated.{result_text}"

    def _handle_disposition(self, action: Action) -> Tuple[float, str]:
        pid = action.patient_id
        disp = action.disposition
        if pid not in self._patients:
            return -0.1, f"Patient {pid} not found."
        if disp not in VALID_DISPOSITIONS:
            return -0.1, f"Invalid disposition '{disp}'."
        correct = self._patients[pid].correct_disposition
        self._dispositions[pid] = disp
        if disp == correct:
            return 0.3, f"Patient {pid}: disposition '{disp}' — CORRECT."
        else:
            msg = f"Patient {pid}: disposition '{disp}' — WRONG (should be '{correct}')."
            if correct == "admit" and disp == "discharge":
                return -0.2, msg + " Dangerous discharge of sick patient!"
            return -0.1, msg

    def _handle_reassess(self, action: Action) -> Tuple[float, str]:
        pid = action.patient_id
        if pid not in self._patients:
            return -0.05, f"Patient {pid} not found for reassessment."
        p = self._patients[pid]
        # Small reward for reassessing patients with pending results
        if self._test_results.get(pid):
            return 0.05, (
                f"Reassessing {pid} with available results: "
                f"{list(self._test_results[pid].keys())}"
            )
        return 0.01, f"Reassessing {pid} — no new results yet."

    def _compute_final_reward(self) -> Tuple[float, Dict, str]:
        """Holistic end-of-episode scoring."""
        breakdown: Dict[str, float] = {}
        patients = list(self._patients.values())

        # 1. Urgency accuracy
        urgency_scores = []
        for p in patients:
            if p.patient_id in self._assigned_urgency:
                s = _urgency_score(self._assigned_urgency[p.patient_id], p.true_urgency)
                urgency_scores.append(s)
            else:
                urgency_scores.append(0.0)
        breakdown["urgency_accuracy"] = sum(urgency_scores) / len(urgency_scores) if urgency_scores else 0.0

        # 2. Test ordering
        test_scores = []
        for p in patients:
            s = _test_score(self._ordered_tests.get(p.patient_id, []), p.required_tests)
            test_scores.append(s)
        breakdown["test_appropriateness"] = sum(test_scores) / len(test_scores) if test_scores else 0.0

        # 3. Disposition
        disp_scores = []
        for p in patients:
            if p.patient_id in self._dispositions:
                s = _disposition_score(self._dispositions[p.patient_id], p.correct_disposition)
                disp_scores.append(s)
            else:
                disp_scores.append(0.0)
        breakdown["disposition_accuracy"] = sum(disp_scores) / len(disp_scores) if disp_scores else 0.0

        # 4. Critical patient safety bonus/penalty
        safety_bonus = 0.0
        for p in patients:
            if p.true_urgency == 1:
                assigned = self._assigned_urgency.get(p.patient_id, 5)
                if assigned == 1:
                    safety_bonus += 0.1
                elif assigned >= 3:
                    safety_bonus -= 0.2  # dangerous under-triage
        breakdown["critical_safety"] = safety_bonus

        # Weighted total
        total = (
            breakdown["urgency_accuracy"] * 0.30 +
            breakdown["test_appropriateness"] * 0.35 +
            breakdown["disposition_accuracy"] * 0.25 +
            min(0.1, max(-0.3, breakdown["critical_safety"]))
        )
        total = max(0.0, min(1.0, total))
        breakdown["final_total"] = total

        msg = (
            f"Episode complete. Urgency={breakdown['urgency_accuracy']:.2f}, "
            f"Tests={breakdown['test_appropriateness']:.2f}, "
            f"Disposition={breakdown['disposition_accuracy']:.2f}, "
            f"Safety={breakdown['critical_safety']:.2f} → Final={total:.2f}"
        )
        return total, breakdown, msg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(self, message: str) -> Observation:
        elapsed = int(time.time() - self._episode_start)
        actions = [
            "assign_urgency(patient_id, urgency_level 1-5)",
            "order_test(patient_id, test_name)",
            "set_disposition(patient_id, admit|discharge|observe|transfer)",
            "reassess(patient_id)",
            "done()"
        ]
        return Observation(
            task_name=self.task.name,
            step=self._step_count,
            patients=list(self._patients.values()),
            ordered_tests=self._ordered_tests,
            test_results=self._test_results,
            assigned_urgency=self._assigned_urgency,
            dispositions=self._dispositions,
            elapsed_minutes=elapsed,
            message=message,
            available_actions=actions,
        )

    # ------------------------------------------------------------------
    # Task grader (standalone, deterministic)
    # ------------------------------------------------------------------

    def grade(self) -> float:
        """Return a deterministic score in [0.0, 1.0] for current state."""
        _, breakdown, _ = self._compute_final_reward()
        return breakdown["final_total"]
