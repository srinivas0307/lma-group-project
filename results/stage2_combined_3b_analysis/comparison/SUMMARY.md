# Fresh Stage 2 parser comparison

Diagnostic end-to-end comparison of different parser methods/adapters on previously inspected tasks; not independent test evidence. Timings include per-task parser/generation/repair but exclude model loading/setup.

- selective: 74 correct, 1 incorrect accepted, 25 withheld; parser calls 98, parser tool attempts 0 (0 failed), total tokens 556724.
- tools: 74 correct, 1 incorrect accepted, 25 withheld; parser calls 110, parser tool attempts 8 (4 failed), total tokens 685765.
