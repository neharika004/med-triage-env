"""
Quick smoke test — verifies the environment works correctly before submission.
Run: python test_env.py
"""

from medical_triage_env import Action, MedicalTriageEnv

def test_task(task_name: str):
    print(f"\n{'='*50}")
    print(f"Testing: {task_name}")
    print(f"{'='*50}")

    env = MedicalTriageEnv(task_name=task_name)
    obs = env.reset()

    print(f"Reset OK — {len(obs.patients)} patient(s) loaded")
    print(f"Patients: {[p.patient_id for p in obs.patients]}")

    # Assign urgency to first patient
    p0 = obs.patients[0]
    action = Action(action_type="assign_urgency", patient_id=p0.patient_id, urgency_level=1)
    obs, reward, done, info = env.step(action)
    print(f"Step 1 (assign_urgency): reward={reward.value:.2f}, msg={reward.message[:60]}")

    # Order a test
    action = Action(action_type="order_test", patient_id=p0.patient_id, test_name="ECG")
    obs, reward, done, info = env.step(action)
    print(f"Step 2 (order_test ECG): reward={reward.value:.2f}, msg={reward.message[:60]}")

    # Set disposition
    action = Action(action_type="set_disposition", patient_id=p0.patient_id, disposition="admit")
    obs, reward, done, info = env.step(action)
    print(f"Step 3 (set_disposition): reward={reward.value:.2f}, msg={reward.message[:60]}")

    # Done
    action = Action(action_type="done")
    obs, reward, done, info = env.step(action)
    print(f"Step 4 (done): reward={reward.value:.2f}, done={done}")

    # Grade
    score = env.grade()
    print(f"Final grade: {score:.4f}")
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    # State check
    state = env.state()
    assert state["task"] == task_name
    print(f"State OK: step={state['step']}, done={state['done']}")

    print(f"✓ {task_name} PASSED")
    return score


if __name__ == "__main__":
    scores = []
    for task in ["triage_easy", "triage_medium", "triage_hard"]:
        try:
            score = test_task(task)
            scores.append(score)
        except Exception as e:
            print(f"✗ {task} FAILED: {e}")
            raise

    print(f"\n{'='*50}")
    print(f"ALL TESTS PASSED")
    print(f"Scores: {[f'{s:.2f}' for s in scores]}")
    print(f"All scores in [0, 1]: {all(0 <= s <= 1 for s in scores)}")
    print(f"{'='*50}")
