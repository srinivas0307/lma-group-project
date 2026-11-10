# np1 offline replay (no-progress stop), from the recorded agent test runs

Exact under greedy decoding: each run is the recorded run cut at the stop point. max_attempts = 3, so k = 3 only relabels REJECT as ESCALATE and saves no calls.

| system | k | stopped | of which recorded REJECT / ACCEPT | repairs lost | R repaired | E escalated | escalations (all) | LLM calls | tokens | tokens saved |
|---|---|---|---|---|---|---|---|---|---|---|
| agent_full | recorded | - | - | - | 58/82 | 8/12 | 17 | 457 | 980554 | - |
| agent_full | 1 | 39 | 23 / 10 | 7 | 53/82 | 12/12 | 50 | 323 | 653643 | 326911 |
| agent_full | 2 | 29 | 22 / 2 | 2 | 56/82 | 12/12 | 41 | 415 | 866846 | 113708 |
| agent_full | 3 | 18 | 18 / 0 | 0 | 58/82 | 12/12 | 35 | 457 | 980554 | 0 |
| agent_feedback | recorded | - | - | - | 51/82 | 10/12 | 32 | 220 | 180032 | - |
| agent_feedback | 1 | 46 | 19 / 5 | 5 | 47/82 | 12/12 | 56 | 145 | 116664 | 63368 |
| agent_feedback | 2 | 28 | 14 / 0 | 0 | 51/82 | 12/12 | 46 | 195 | 158718 | 21314 |
| agent_feedback | 3 | 9 | 9 / 0 | 0 | 51/82 | 11/12 | 41 | 220 | 180032 | 0 |
| agent_naive | recorded | - | - | - | 50/82 | 12/12 | 55 | 222 | 150347 | - |
| agent_naive | 1 | 45 | 0 / 0 | 0 | 50/82 | 12/12 | 55 | 142 | 97497 | 52850 |
| agent_naive | 2 | 35 | 0 / 0 | 0 | 50/82 | 12/12 | 55 | 187 | 127617 | 22730 |
| agent_naive | 3 | 0 | 0 / 0 | 0 | 50/82 | 12/12 | 55 | 222 | 150347 | 0 |
