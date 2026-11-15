# Combined 3B gold-contract diagnostic

Oracle diagnostic on previously inspected tasks. Gold replaces the contract in generation and verification; initial programs can differ. This is the total contract-source effect, not a repair-only ablation or guaranteed upper bound. Setup/model loading excluded from task time; gold has zero parser cost.

- gold: 84 correct, 0 incorrect accepted, 16 withheld; total tokens 327153 (parser 0).
- selective: 74 correct, 1 incorrect accepted, 25 withheld; total tokens 556724 (parser 177514).
- tools: 74 correct, 1 incorrect accepted, 25 withheld; total tokens 685765 (parser 258796).
- Gold versus selective: 13 tasks correct only with gold; 3 correct only with predicted contracts.
- Gold versus tools: 13 tasks correct only with gold; 3 correct only with predicted contracts.
