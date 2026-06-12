# Stage4 Local Vision App

Stage4 integrates the existing Stage3 Task9 dynamic face effects into a local desktop application and CLI. It does not retrain models, download datasets, or move Stage2/Stage3 files.

## Entry Points

```bash
python stage-4/code/stage4_run_cli.py --check-env
python stage-4/code/stage4_run_cli.py --video stage-3/reports/task9/assets/videos/task9_dynamic_effects_demo.mp4
python stage-4/code/stage4_desktop_app.py
```

## Dependencies

Install the local runtime dependencies in the project conda environment:

```bash
pip install -r stage-4/requirements-stage4.txt
```

## Outputs

- Summary: `stage-4/reports/summaries/stage4_integration_summary.json`
- Report: `stage-4/reports/stage4_project_integration_report.md`
- Exported videos: `stage-4/reports/assets/videos/`
