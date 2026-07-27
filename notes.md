uv run python src/glm/queue_statistical_experiments.py --dry-run    # preview the 20 commands
uv run python src/glm/queue_statistical_experiments.py              # actually queue them
dvc queue status                                                    # inspect the queue
dvc exp run --run-all                                               # execute sequentially (single GPU — don't pass -j)

- Follow live output from a running queued task: dvc queue logs -f <task-id> (e.g. dvc queue logs -f 45f9002), add -v for verbose too.
- For a directly-run (non-queued) dvc exp run or dvc repro, just add -v (e.g. dvc exp run -v). You can also export DVC_LOGLEVEL=debug for maximum verbosity from any dvc command.