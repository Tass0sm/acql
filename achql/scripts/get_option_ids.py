import argparse
import mlflow

def get_nested_runs(parent_run_id, experiment_id=None):
    filter_str = f"tags.mlflow.parentRunId = '{parent_run_id}'"
    return mlflow.search_runs(
        experiment_ids=[experiment_id] if experiment_id else None,
        filter_string=filter_str,
        output_format="list"
    )

def main(spec, alg, experiment_id=None):
    mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")

    parent_filter = f"tags.spec = '{spec}' and tags.alg = '{alg}'"
    parent_runs = mlflow.search_runs(
        experiment_ids=[experiment_id] if experiment_id else None,
        filter_string=parent_filter,
        output_format="list"
    )

    for run in parent_runs:
        print(f"\nParent Run: {run.info.run_id}")
        print(f"  - Spec: {run.data.tags.get('spec')}")
        print(f"  - Alg:  {run.data.tags.get('alg')}")
        print(f"  - Seed: {run.data.params.get('seed')}")  # <- Added line

        child_runs = get_nested_runs(run.info.run_id, experiment_id)
        for child in child_runs:
            print(f"    Child Run: {child.info.run_id}")
            print(f"      - Status: {child.info.status}")
            print(f"      - Metrics: {child.data.metrics}")
            print(f"      - Params: {child.data.params}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query MLflow runs with nested children.")
    parser.add_argument("--spec", required=True, help="Task specification tag value.")
    parser.add_argument("--alg", required=True, help="Algorithm tag value.")
    parser.add_argument("--experiment-id", type=str, help="Optional MLflow experiment ID.")
    args = parser.parse_args()

    main(args.spec, args.alg, args.experiment_id)
