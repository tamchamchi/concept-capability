# Run the evaluator pipeline on Google Colab

The pipeline generates all required data locally in the Colab runtime, validates
dataset leakage and balance, evaluates the rule baseline, trains the dual-head
CNN on GPU, and saves its checkpoint, metrics, CSV log, and training curves.

## 1. Enable a GPU

In Colab select **Runtime → Change runtime type → T4 GPU**.

## 2. Make the repository available

Clone the repository if it has a remote URL:

```python
!git clone YOUR_REPOSITORY_URL /content/concept-capability
%cd /content/concept-capability
```

Alternatively upload/extract the repository at
`/content/concept-capability`, then change into that directory.

## 3. Run end to end

Fast local runtime storage is recommended for the many small PNG files:

```python
!bash scripts/run_evaluator_colab.sh /content/concept-capability-data
```

The command is resumable: valid existing datasets and accepted model metrics are
reused. Add `--retrain` when invoking the Python pipeline directly to replace the
model run.

## 4. Persist the result to Google Drive

Mount Drive and pass an archive destination:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!bash scripts/run_evaluator_colab.sh \
  /content/concept-capability-data \
  /content/drive/MyDrive/concept-capability/evaluator-pipeline.tar.gz
```

Important outputs before archiving:

```text
/content/concept-capability-data/evaluator/model/evaluator.pt
/content/concept-capability-data/evaluator/model/metrics.json
/content/concept-capability-data/evaluator/model/history.csv
/content/concept-capability-data/evaluator/model/training_curves.png
/content/concept-capability-data/evaluator/rule_based_metrics.json
/content/concept-capability-data/evaluator/pipeline_report.json
```

To validate data without training:

```python
!python scripts/run_evaluator_pipeline.py \
  --data-root /content/concept-capability-data \
  --validate-only
```
