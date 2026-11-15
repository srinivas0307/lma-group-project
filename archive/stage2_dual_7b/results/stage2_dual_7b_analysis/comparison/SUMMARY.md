# Fresh Stage 2 parser comparison

Diagnostic end-to-end comparison of different parser methods/adapters on previously inspected tasks; not independent test evidence. Timings include per-task parser/generation/repair but exclude model loading/setup.

- selective: 78 correct, 1 incorrect accepted, 21 withheld; parser calls 98, parser tool attempts 0 (0 failed), total tokens 513385.
- tools: 78 correct, 1 incorrect accepted, 21 withheld; parser calls 110, parser tool attempts 8 (4 failed), total tokens 688063.
