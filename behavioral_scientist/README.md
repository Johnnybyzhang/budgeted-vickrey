Behavioral Scientist Track — oTree + LLM Artifacts

Goal
- Document the human-subjects interface and session configuration; package the oTree app and provide screenshots and LLM prompt artifacts.

Screenshots
- Under [Screenshots](./screenshots)

LLM Prompts
- prompts and transcripts under behavioral_scientist/llm/:
  - prompts.txt — the exact prompts used
  - transcript.md — discussion trace with timestamps
  - settings.json — model/version/temperature and other parameters

Deployment Steps
- Create a new oTree project, add app, and configure session defaults matching T, B0, r, value range.
- Test locally with otree devserver; deploy to oTree Hub or a VM (Used).

Session Configuration
- Players: 2 (extendable)
- Values: Uniform[60,100]
- Initial budget B0>100; refill r∈(0,60)
- Bid constraint: b≤current budget
- Payments: second price; ties split

Ethics Note
- Ensure consent language explains budget constraints and payments.
- Avoid deception; disclose that bids are capped by remaining budget.
- Anonymize any stored identifiers and follow IRB guidance.

Generated data following the observed behaviour of class assignment at [CSV File](./all_apps_wide-2025-09-11.csv)
