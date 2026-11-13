# Independent development snapshots

- [Phase A](phase_a): three-system generator-prompt comparison.
- [Phase B](phase_b): four-system repair-prompt comparison, including its own controls.

Each child contains system folders and the original bundle metadata, logs and metrics. The original uploaded ZIPs remain in `kaggle_outputs/stage2/dev/`.

Pass a snapshot directory, not this parent, to the metrics script:

```bash
.venv/bin/python eval/stage2_metrics.py results/stage2_prompt_dev/phase_b dev
```

See [the complete artifact index](../../../results/STAGE2_DEV_RESULTS.md).
