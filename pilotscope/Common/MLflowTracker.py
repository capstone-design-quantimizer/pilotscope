"""
MLflow Tracker for PilotScope
==============================
Wrapper class to integrate MLflow experiment tracking with minimal code changes.

Features:
- Automatic experiment tracking (parameters, metrics, models)
- Run management (training + testing in same run)
- Model artifact logging
- Best model retrieval
"""

import mlflow
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    PilotScope용 MLflow 추적 래퍼
    기존 코드 변경을 최소화하면서 MLflow 통합
    """

    def __init__(self, tracking_uri: str = None, experiment_name: str = "pilotscope"):
        """
        Args:
            tracking_uri: MLflow 저장 경로. None이면 mlruns/ 사용
            experiment_name: 실험 이름 (알고리즘별로 구분 권장)
        """
        # Set tracking URI (default: mlruns/ in project root)
        if tracking_uri is None:
            # Use mlruns directory in project root
            project_root = Path(__file__).parent.parent.parent.resolve()
            tracking_uri = str((project_root / "mlruns").absolute())

        # MLflow expects absolute path without file:// prefix for local paths
        mlflow.set_tracking_uri(tracking_uri)

        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                # Create new experiment
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
            elif experiment.lifecycle_stage == "deleted":
                # Experiment was deleted - restore or recreate
                logger.warning(f"Experiment '{experiment_name}' is deleted. Restoring...")
                try:
                    # Try to restore
                    client = mlflow.tracking.MlflowClient()
                    client.restore_experiment(experiment.experiment_id)
                    logger.info(f"Restored deleted experiment: {experiment_name}")
                    experiment_id = experiment.experiment_id
                except Exception as restore_error:
                    # If restore fails, permanently delete and recreate
                    logger.warning(f"Could not restore, recreating: {restore_error}")
                    try:
                        client.delete_experiment(experiment.experiment_id)
                        experiment_id = mlflow.create_experiment(experiment_name)
                        logger.info(f"Recreated experiment: {experiment_name} (ID: {experiment_id})")
                    except:
                        # Last resort: use a new name with timestamp
                        import time
                        new_name = f"{experiment_name}_{int(time.time())}"
                        experiment_id = mlflow.create_experiment(new_name)
                        logger.info(f"Created new experiment with timestamp: {new_name}")
                        experiment_name = new_name
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")

            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"Failed to set experiment {experiment_name}: {e}")
            # Fallback to default experiment
            mlflow.set_experiment("Default")
            experiment_name = "Default"

        self.experiment_name = experiment_name
        self.active_run = None
        self.run_id = None
        self._enabled = True  # Can be disabled for debugging
        self._dataset_info = {}  # Store dataset info for test logging

    def start_training(self, algo_name: str, dataset: str, params: Dict, db_name: str = None, workload: str = None, num_queries: int = None) -> Optional[str]:
        """
        학습 시작시 호출 - 새로운 MLflow run 생성

        Args:
            algo_name: 알고리즘 이름 (e.g., "mscn", "lero")
            dataset: 데이터셋 이름 (e.g., "stats_tiny_custom")
            params: 하이퍼파라미터 dict
            db_name: 실제 DB 이름 (e.g., "stats_tiny")
            workload: 워크로드 이름 (e.g., "custom", "default")
            num_queries: 학습 쿼리 수

        Returns:
            run_id: MLflow run ID (나중에 재연결할 때 사용)
        """
        if not self._enabled:
            return None

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{algo_name}_{dataset}_{timestamp}"

            self.active_run = mlflow.start_run(run_name=run_name)
            self.run_id = self.active_run.info.run_id

            # Store dataset info for later use
            self._dataset_info = {
                "dataset": dataset,
                "db_name": db_name or dataset,
                "workload": workload or "default",
                "num_training_queries": num_queries
            }

            # Log basic parameters
            mlflow.log_params({
                "algorithm": algo_name,
                "dataset": dataset,
                **self._flatten_params(params)
            })

            # Log dataset info as tags
            mlflow.set_tags({
                "dataset.name": dataset,
                "dataset.db": db_name or dataset,
                "dataset.workload": workload or "default",
                "dataset.context": "training"
            })

            # Log dataset for "Datasets used" column (MLflow 2.3+)
            try:
                import pandas as pd
                import warnings
                # Suppress MLflow's integer column warning (it's just a warning, not an error)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*integer column.*")
                    warnings.filterwarnings("ignore", message=".*UCVolumeDatasetSource.*")
                    warnings.filterwarnings("ignore", message=".*LocalArtifactDatasetSource.*")

                    # Create metadata DataFrame with workload info
                    dataset_df = pd.DataFrame({
                        "dataset": [dataset],
                        "db": [db_name or dataset],
                        "workload": [workload or "default"],
                        "num_queries": [num_queries if num_queries else 0]
                    })
                    # Log as MLflow dataset
                    dataset_source = mlflow.data.from_pandas(
                        dataset_df,
                        source=dataset,
                        name=dataset,
                        targets=None
                    )
                    mlflow.log_input(dataset_source, context="training")
            except Exception as e:
                logger.debug(f"Could not log dataset to Datasets column: {e}")

            # Log system info as tags
            mlflow.set_tags({
                "stage": "training",
                "started_at": datetime.now().isoformat(),
                "mlflow.note.content": f"Algorithm: {algo_name}, Dataset: {dataset}"
            })

            logger.info(f"Started MLflow run: {run_name} (ID: {self.run_id})")
            return self.run_id

        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            self._enabled = False
            return None

    def log_training_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        학습 메트릭 로깅 (epoch별 loss 등)

        Args:
            metrics: 메트릭 dict (e.g., {"train_loss": 0.5})
            step: Epoch 번호 (선택사항)
        """
        if not self._enabled or self.active_run is None:
            return

        try:
            if step is not None:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=step)
            else:
                mlflow.log_metrics(metrics)
        except Exception as e:
            logger.error(f"Failed to log training metrics: {e}")

    def log_model_metadata(self, model_path: str, model_metadata: Dict[str, Any] = None, algorithm: str = None):
        """
        모델 파일과 메타데이터를 MLflow Model로 저장 (Logged models 컬럼에 표시)

        Args:
            model_path: 모델 파일 경로 (기존 저장 방식 유지)
            model_metadata: 추가 메타데이터 (선택사항)
            algorithm: 알고리즘 이름 (선택사항)
        """
        if not self._enabled or self.active_run is None:
            return

        try:
            model_path = Path(model_path)
            model_name = model_metadata.get('model_id', model_path.stem) if model_metadata else model_path.stem

            # Log model file as artifact (legacy)
            if model_path.exists():
                mlflow.log_artifact(str(model_path), "model_files")
                logger.info(f"Logged model artifact: {model_path}")

            # Log metadata JSON if exists
            metadata_path = Path(f"{model_path}.json")
            if metadata_path.exists():
                mlflow.log_artifact(str(metadata_path), "model_files")
                logger.info(f"Logged metadata artifact: {metadata_path}")

            # Log as MLflow pyfunc model (for "Logged models" column)
            try:
                # Create a simple Python function wrapper
                class PilotScopeModelWrapper(mlflow.pyfunc.PythonModel):
                    """Wrapper for PilotScope models to enable MLflow tracking"""

                    def __init__(self, model_path, metadata):
                        self.model_path = str(model_path)
                        self.metadata = metadata or {}

                    def predict(self, context, model_input):
                        """Placeholder predict method"""
                        return {"error": "Direct prediction not supported. Load model using PilotScope."}

                # Create wrapper instance
                wrapper = PilotScopeModelWrapper(model_path, model_metadata)

                # Log as pyfunc model
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=wrapper,
                    artifacts={"model_file": str(model_path)} if model_path.exists() else None,
                    registered_model_name=None,  # Don't register yet
                    metadata={
                        "model_name": model_name,
                        "algorithm": algorithm or "unknown",
                        "model_path": str(model_path),
                        **(model_metadata or {})
                    }
                )

                logger.info(f"Logged model to MLflow Models: {model_name}")

            except Exception as e:
                logger.warning(f"Could not log as MLflow pyfunc model: {e}")
                # Fallback: log model info as JSON
                mlflow.log_dict(
                    {
                        "model_name": model_name,
                        "algorithm": algorithm or "unknown",
                        "model_path": str(model_path),
                        "metadata": model_metadata or {}
                    },
                    "model_info.json"
                )

            # Set model tags for UI
            mlflow.set_tag("mlflow.model.type", algorithm or "custom")
            mlflow.set_tag("model_name", model_name)

            # Log additional metadata as params/tags
            if model_metadata:
                for k, v in model_metadata.items():
                    try:
                        mlflow.log_param(f"model_{k}", v)
                    except:
                        pass  # Skip if parameter already exists

        except Exception as e:
            logger.error(f"Failed to log model: {e}")

    def log_test_results(self, test_metrics: Dict[str, float], test_dataset: str = None, num_test_queries: int = None):
        """
        테스트 결과 로깅 (같은 run에 추가)

        Args:
            test_metrics: 테스트 메트릭 dict
            test_dataset: 테스트 데이터셋 이름 (선택사항)
            num_test_queries: 테스트 쿼리 수 (선택사항)
        """
        if not self._enabled or self.active_run is None:
            return

        try:
            # Prefix with "test_" to distinguish from training metrics
            prefixed_metrics = {}
            for k, v in test_metrics.items():
                # Skip if already has test_ prefix
                key = k if k.startswith("test_") else f"test_{k}"
                prefixed_metrics[key] = v

            mlflow.log_metrics(prefixed_metrics)

            # Update tags
            tags = {
                "stage": "tested",
                "tested_at": datetime.now().isoformat()
            }
            if test_dataset:
                tags["test_dataset"] = test_dataset
                # Also log as dataset tag for MLflow UI
                tags["dataset.test.name"] = test_dataset

                # Log test dataset for "Datasets used" column
                try:
                    import pandas as pd
                    import warnings
                    # Suppress MLflow's integer column warning
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*integer column.*")
                        warnings.filterwarnings("ignore", message=".*UCVolumeDatasetSource.*")
                        warnings.filterwarnings("ignore", message=".*LocalArtifactDatasetSource.*")

                        # Use stored dataset info from training
                        db_name = self._dataset_info.get("db_name", test_dataset)
                        workload = self._dataset_info.get("workload", "default")

                        dataset_df = pd.DataFrame({
                            "dataset": [test_dataset],
                            "db": [db_name],
                            "workload": [workload],
                            "num_queries": [num_test_queries if num_test_queries else 0]
                        })
                        # Log as MLflow dataset
                        dataset_source = mlflow.data.from_pandas(
                            dataset_df,
                            source=test_dataset,
                            name=test_dataset,
                            targets=None
                        )
                        mlflow.log_input(dataset_source, context="testing")
                except Exception as e:
                    logger.debug(f"Could not log test dataset to Datasets column: {e}")

            mlflow.set_tags(tags)

            logger.info(f"Logged test results: {prefixed_metrics}")

        except Exception as e:
            logger.error(f"Failed to log test results: {e}")

    def end_run(self, status: str = "FINISHED"):
        """
        실험 종료

        Args:
            status: Run 상태 ("FINISHED", "FAILED", "KILLED")
        """
        if not self._enabled or self.active_run is None:
            return

        try:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run: {self.run_id} (status: {status})")
            self.active_run = None
            self.run_id = None
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")

    def disable(self):
        """MLflow 추적 비활성화 (디버깅용)"""
        self._enabled = False
        logger.info("MLflow tracking disabled")

    def enable(self):
        """MLflow 추적 활성화"""
        self._enabled = True
        logger.info("MLflow tracking enabled")

    @staticmethod
    def _flatten_params(params: Dict, parent_key: str = "", sep: str = "_") -> Dict:
        """
        중첩된 dict를 평탄화 (MLflow는 nested params를 지원하지 않음)

        Example:
            {"model": {"lr": 0.01}} -> {"model_lr": 0.01}
        """
        items = []
        for k, v in params.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflowTracker._flatten_params(v, new_key, sep=sep).items())
            else:
                # Convert to JSON-serializable type
                if isinstance(v, (int, float, str, bool)) or v is None:
                    items.append((new_key, v))
                else:
                    items.append((new_key, str(v)))
        return dict(items)

    @classmethod
    def get_best_run(cls, experiment_name: str, metric: str = "test_total_time",
                     ascending: bool = True) -> Optional[Dict]:
        """
        최고 성능 run 찾기

        Args:
            experiment_name: 실험 이름
            metric: 비교 메트릭 (e.g., "test_total_time", "test_average_time")
            ascending: True면 작은 값이 좋음 (시간), False면 큰 값이 좋음 (정확도)

        Returns:
            Dict with run info or None if no runs found
        """
        try:
            exp = mlflow.get_experiment_by_name(experiment_name)
            if exp is None:
                logger.warning(f"Experiment not found: {experiment_name}")
                return None

            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string=f"metrics.{metric} > 0",  # Filter out failed runs
                order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
                max_results=1
            )

            if len(runs) == 0:
                logger.warning(f"No runs found in experiment: {experiment_name}")
                return None

            best_run = runs.iloc[0]

            return {
                "run_id": best_run.run_id,
                "run_name": best_run.tags.get("mlflow.runName", ""),
                "metric_name": metric,
                "metric_value": best_run[f"metrics.{metric}"],
                "params": {k.replace("params.", ""): v for k, v in best_run.items()
                          if k.startswith("params.")},
                "artifact_uri": best_run.artifact_uri
            }

        except Exception as e:
            logger.error(f"Failed to get best run: {e}")
            return None

    @classmethod
    def list_runs(cls, experiment_name: str, limit: int = 10) -> list:
        """
        실험의 run 목록 조회

        Args:
            experiment_name: 실험 이름
            limit: 최대 결과 수

        Returns:
            List of run dicts
        """
        try:
            exp = mlflow.get_experiment_by_name(experiment_name)
            if exp is None:
                return []

            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=limit
            )

            return runs.to_dict('records')

        except Exception as e:
            logger.error(f"Failed to list runs: {e}")
            return []

    @classmethod
    def download_model_artifact(cls, run_id: str, dest_dir: str = None) -> Optional[str]:
        """
        MLflow run에서 모델 artifact를 다운로드

        Args:
            run_id: MLflow run ID
            dest_dir: 저장할 디렉토리 (None이면 임시 디렉토리)

        Returns:
            다운로드된 모델 파일 경로 또는 None
        """
        try:
            import tempfile
            import shutil

            # Get artifact URI
            run = mlflow.get_run(run_id)
            artifact_uri = run.info.artifact_uri

            # Determine destination
            if dest_dir is None:
                dest_dir = tempfile.mkdtemp(prefix="pilotscope_model_")
            else:
                os.makedirs(dest_dir, exist_ok=True)

            # Download artifacts from "model_files" directory
            client = mlflow.tracking.MlflowClient()
            try:
                artifacts = client.list_artifacts(run_id, "model_files")
                if not artifacts:
                    logger.warning(f"No model_files artifacts found in run {run_id}")
                    return None

                # Download all artifacts in model_files
                for artifact in artifacts:
                    artifact_path = client.download_artifacts(run_id, artifact.path, dest_dir)
                    logger.info(f"Downloaded artifact: {artifact_path}")

                # Find the main model file (not .json)
                downloaded_files = list(Path(dest_dir).glob("model_files/*"))
                model_files = [f for f in downloaded_files if not f.name.endswith('.json')]

                if model_files:
                    return str(model_files[0])
                else:
                    logger.warning("No model file found (only metadata)")
                    return None

            except Exception as e:
                logger.warning(f"Could not download from model_files: {e}")
                # Fallback: try to download from root
                try:
                    artifact_path = client.download_artifacts(run_id, "", dest_dir)
                    return artifact_path
                except:
                    return None

        except Exception as e:
            logger.error(f"Failed to download model artifact: {e}")
            return None

    def save_model_artifact(self, model_path: str, model_id: str = None,
                           metadata: Dict = None) -> bool:
        """
        현재 run에 모델을 artifact로 저장

        Args:
            model_path: 저장할 모델 파일 경로
            model_id: 모델 ID (타임스탬프 기반)
            metadata: 추가 메타데이터

        Returns:
            성공 여부
        """
        if not self._enabled or self.active_run is None:
            logger.warning("Cannot save model: no active MLflow run")
            return False

        try:
            model_path = Path(model_path)

            # Save model file as artifact
            if model_path.exists() and model_path.is_file():
                mlflow.log_artifact(str(model_path), "model_files")
                logger.info(f"Saved model artifact: {model_path}")
            elif model_path.exists() and model_path.is_dir():
                # If it's a directory, log all files in it
                for file in model_path.iterdir():
                    if file.is_file():
                        mlflow.log_artifact(str(file), "model_files")
                logger.info(f"Saved model directory artifacts: {model_path}")
            else:
                logger.warning(f"Model path does not exist: {model_path}")
                return False

            # Save metadata JSON if exists
            metadata_path = Path(f"{model_path}.json")
            if metadata_path.exists():
                mlflow.log_artifact(str(metadata_path), "model_files")
                logger.info(f"Saved metadata artifact: {metadata_path}")

            # Log model_id as parameter for easy retrieval
            if model_id:
                try:
                    mlflow.log_param("model_id", model_id)
                except:
                    # Parameter might already exist
                    mlflow.set_tag("model_id", model_id)

            # Log metadata as JSON
            if metadata:
                mlflow.log_dict(metadata, "model_metadata.json")

            return True

        except Exception as e:
            logger.error(f"Failed to save model artifact: {e}")
            return False
