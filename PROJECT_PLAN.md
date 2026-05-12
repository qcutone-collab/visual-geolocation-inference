# Final Project Plan

## Project Goal
Deliver a country geolocation web app that is reliable, explainable, and evaluated with clear metrics, with the strongest accuracy achievable on the available dataset.

## Success Criteria
- Primary metric: top-1 country accuracy on held-out test set
- Secondary metrics: top-3 accuracy, per-country precision/recall, confusion matrix quality
- Product metric: Streamlit app returns prediction + confidence + top-3 countries
- Engineering metric: reproducible training and inference pipeline with saved artifacts

## Phase Plan (Now -> Final Delivery)

### Phase 1: Baseline Stabilization (Current Week)
- Lock consistent train/inference feature pipeline
- Verify data integrity (class balance, corrupt images, duplicate leakage)
- Re-run baseline and document metrics
- Deliverable: `models/country_model.pkl` + baseline evaluation report

### Phase 2: Data Quality & Splits (Week 1)
- Add strict train/val/test split policy (avoid location leakage where possible)
- Balance classes with sampling strategy and minimum image thresholds
- Add data-clean checks (resolution floor, blur/outlier filtering)
- Deliverable: cleaned dataset manifest and reproducible split script

### Phase 3: Model Upgrades (Week 2)
- Compare multiple backbones/features:
  - CNN embeddings + classical ML
  - Fine-tuned transfer model (if compute allows)
  - Ensemble of visual + handcrafted cues
- Track each experiment in a simple results table
- Deliverable: best candidate model with metrics and ablation notes

### Phase 4: Error-Driven Improvement (Week 3)
- Analyze confusion pairs (countries commonly mixed up)
- Add targeted features/augmentations for hard classes
- Improve calibration of confidence scores
- Deliverable: improved model and error analysis summary

### Phase 5: App & UX Finalization (Week 4)
- Integrate best model into `app.py`
- Add user-facing confidence text and top-3 explanation
- Add graceful handling for unreadable/unsupported images
- Deliverable: demo-ready Streamlit app

### Phase 6: Final Validation & Packaging (Final Week)
- Final test run with frozen model and dataset split
- Save artifacts, scripts, and reproducibility instructions
- Prepare final presentation/demo script
- Deliverable: final repo, final metrics sheet, demo checklist

---

# In-Class Working Plan (Template Used Every Session)

## Session Header
- Date:
- Session #:
- Duration:
- Main Objective:

## 1) Planned Tasks (Before Class)
- Task A:
- Task B:
- Task C:

## 2) In-Class Execution Plan (Time-Boxed)
- 0-15 min: quick status + unblock issues
- 15-45 min: implement highest-priority model/data task
- 45-70 min: run experiment/evaluation
- 70-85 min: interpret results + decide next action
- 85-90 min: log outcomes and update next session tasks

## 3) Deliverables by End of Session
- Code changes completed:
- Experiment(s) run:
- Metric(s) recorded:
- Files/artifacts produced:

## 4) Risks / Blockers
- Blocker 1:
- Blocker 2:
- Mitigation plan:

## 5) Next Session Prep
- Immediate follow-up task:
- Required files/data ready:
- Clear success target for next session:

---

# In-Class Working Plan (Next 6 Sessions)

## Session 1 - Data Audit & Split Integrity
- Objective: establish trustworthy train/val/test process
- Focus: class counts, duplicates, leakage checks, split script
- End output: dataset report + saved split indexes

## Session 2 - Baseline Rebuild
- Objective: rebuild baseline with clean split
- Focus: train current pipeline, capture top-1/top-3/confusion matrix
- End output: baseline metrics snapshot

## Session 3 - Model Experiment Round 1
- Objective: test stronger feature/model candidates
- Focus: CNN embedding variants and classifier sweep
- End output: experiment comparison table

## Session 4 - Error Analysis & Targeted Fixes
- Objective: reduce biggest confusion clusters
- Focus: hard-country pairs, augmentation/feature tweaks
- End output: improved per-class scores

## Session 5 - Model Freeze & App Integration
- Objective: lock best model and connect to Streamlit
- Focus: inference consistency, confidence display, robustness checks
- End output: demo-capable app build

## Session 6 - Final Validation & Demo Prep
- Objective: final metrics and presentation readiness
- Focus: full end-to-end test, reproducibility run, demo script
- End output: final metrics, final checklist, submission package
