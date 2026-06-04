# API Endpoints

| Method | Endpoint                                    | Description                                     |
| ------ | ------------------------------------------- | ----------------------------------------------- |
| GET    | `/`                                         | API root with service info                      |
| GET    | `/api/health`                               | Health check with model status and memory usage |
| GET    | `/api/model-types`                          | Get available model types                       |
| GET    | `/api/models`                               | Get model type values                           |
| GET    | `/api/weight-files`                         | Get available model weight files                |
| GET    | `/api/config-files`                         | Get available dataset config files              |
| POST   | `/api/pdf-extract`                          | Extract PDF pages to PNG images                 |
| POST   | `/api/detect`                               | Run object detection on uploaded image          |
| POST   | `/api/export/excel`                         | Export detection/pipeline results to Excel      |
| POST   | `/api/pipeline/merge`                       | Run multi-sheet merge engine on completed jobs  |
| GET    | `/api/results/{result_id}`                  | Fetch a previously detected result              |
| PATCH  | `/api/results/{result_id}/objects/{obj_id}` | Update a detected object                        |
| POST   | `/api/results/{result_id}/objects`          | Create a new object                             |
| DELETE | `/api/results/{result_id}/objects/{obj_id}` | Delete a detected object                        |
| POST   | `/api/pipeline/jobs`                        | Start a new pipeline job                        |
| GET    | `/api/pipeline/jobs/{job_id}`               | Get pipeline job status and details             |
| GET    | `/api/pipeline/jobs/{job_id}/stage-status`  | Get status of all pipeline stages               |
| POST   | `/api/pipeline/jobs/{job_id}/resume-from/{stage}` | Resume pipeline from a specific stage       |
| GET    | `/api/pipeline/jobs/{job_id}/review-state`  | Get pipeline review state                       |
| PUT    | `/api/pipeline/jobs/{job_id}/review-state`  | Update pipeline review state                    |
| GET    | `/api/pipeline/jobs/{job_id}/review-workspace` | Get pipeline review workspace               |
| PUT    | `/api/pipeline/jobs/{job_id}/review-workspace` | Update pipeline review workspace            |
| POST   | `/api/pipeline/jobs/{job_id}/review-workspace/recompute` | Recompute pipeline review workspace    |
| POST   | `/api/pipeline/jobs/{job_id}/review-workspace/commit` | Commit pipeline review workspace          |
| GET    | `/api/pipeline/jobs/{job_id}/reviewed-graph`| Get reviewed graph from pipeline                |
| GET    | `/api/pipeline/jobs/{job_id}/reviewed-qa`   | Get reviewed QA report from pipeline            |
| PUT    | `/api/pipeline/jobs/{job_id}/artifacts/{artifact_name}` | Update pipeline artifact              |
| GET    | `/api/pipeline/jobs/{job_id}/artifacts/{artifact_name}` | Get pipeline artifact                 |