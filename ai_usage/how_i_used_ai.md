# How I used AI on this project

I built this take-home with Claude Code (Anthropic) from 14 to 17 November 2026. The full chat logs are in
[chat_logs/](chat_logs/), one file per day: my messages, Claude's replies and one line per tool call. Tool outputs,
the model's internal reasoning and automated notices are left out, and my email address is redacted.

- **Me:** I set the scope and the quality bar, answered open questions, pushed back when results looked too good, and
  made every keep-or-drop call.
- **Claude:** it planned, wrote the code, tests and reports, ran everything, and reported back with numbers.
- **Commits:** nothing was committed without my say-so.

Numbers below come from `docs/CHANGELOG.md`, `docs/PROJECT_HISTORY_AND_METRICS.md`, `model/artifacts/` and
`docs/VISION_REPORT.md`.

## How I worked

1. **Plan before code.** Every stage started with a written plan I corrected, often more than once.
2. **Set the rule before seeing the result.** The drive-model retrain, the fresh test and the vision confirmation rule
   each had their pass rule written down first.
3. **Treat a perfect score as a bug.**
4. **Write down what is weak:** decisions made after seeing results, small samples, and what the simulator makes easy.

## Key decisions and how they paid off

| Decision | Why | Result |
|---|---|---|
| Remove belt damage as a sensor fault | Public images have no link to sensors; any agreement would be wired in by me | Vision stays honest: public dataset, not linked to sensors |
| Vision deferred, not dropped (placeholders first) | The dataset was still downloading | The sensor side never waited; vision plugged into existing tables and endpoints |
| Alert target: at most 1 to 2 false in 20 | "1 false alarm per zone per month" meant 19 of 20 alerts false | 0 false alerts on 32 faults in the fresh test |
| "How do we know a prediction was right?" | A call counted as right just for firing on a degrading part | Strict scoring plus a live Track record; found the drive alert closing 3 h before the failure, which was fixed |
| 72 h failure-call window | The safer side | Chosen after seeing test results, so it loosens the claim (reported as such) |
| The first evaluation was perfect, so I made the data harder: random faults, random noise, drift, shared noise | The model had learned one fault recipe and broke under noise (591 false alerts at 8x) | Test precision fell to 0.38, then wider validation and drift-robust features brought it back to 8 of 10 caught, 0 false |
| Retrain the drive model, keep only if 3 of 6 on validation | It caught 1 of 6 | 4 of 6, 0 false; kept after one attempt |
| Freeze the models, then a fresh test scored once (seeds 501 to 504) | The old test had been used for diagnosis | Bearings 16 of 20, drive 6 of 12, 0 false alerts; criteria met |
| One tab per question, a tour, alerts that say why | The single page was overwhelming | Each required item has its own view |

## Vision

- **Dataset:** BeltCrack9kd. Before any training I asked for an audit (split, near-duplicates, box sizes, licence),
  and I rejected the first training plan until it used the official split and a clear threshold rule.
- **Training my own detector failed.** I tried a replication of the paper's BeltCrackDet, then a smaller YOLO11s. My
  PC did not have the compute and became unusable under the load. Two runs on a Colab T4 stopped when compute credits
  ran out, on two different accounts. BeltCrackDet alone needed about 74 GPU-hours.
- **I switched to a pretrained YOLOv8** by Julián Neira Nova (AGPL-3.0, credited) and evaluated it properly: 26.9%
  mAP50 on the test split, in line with the paper's models.
- **The looping belt:** detections are placed by belt coordinate and merged within 2 m; splices go to an inspect list.
  The checks caught three problems on the way:
  - clean images shown zoomed made the model fire on 10 of 16 of them;
  - a lap-ordering bug in the simulator;
  - the "2 laps" rule removed no false hits.
- **Lap-ratio rule:** I replaced the 2-lap rule with one chosen on a separate run (seed 2027), not the demo. False
  confirmed defects went from 4 to 0, with the same 3 of 4 cracks found. The margin is thin, and I said so.

## What did not pay off

- **Training a detector:** it cost time and compute on three platforms without a finished model. I should have
  checked the compute cost first.
- **Choices made after seeing results** (the 72 h window, the cautious threshold rule) are weaker than rules set in
  advance.
- **The drive model** is still weak evidence: 6 of 12 on the fresh test.

## Keeping the AI honest

- **Tests guard the rules:** no ground truth in the live path, frozen model hashes, test data scored once, and demo
  sensor files byte-identical after vision was added.
- **Mistakes were written up when found,** not hidden. Examples: a commit made while a test was failing, the
  lap-ordering bug, and a stale browser cache that briefly faked a clean check.
- **No false claims:** Claude declined to write in the README that I trained the detector. I gave my real account
  instead.
