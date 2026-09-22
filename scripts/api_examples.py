import json


def main() -> None:
    examples = {
        "health": {
            "method": "GET",
            "url": "http://127.0.0.1:8000/health",
        },
        "assess": {
            "method": "POST",
            "url": "http://127.0.0.1:8000/assess",
            "json": {
                "csv_path": "data/india/india_regions.csv",
            },
        },
        "scenario": {
            "method": "POST",
            "url": "http://127.0.0.1:8000/scenario",
            "json": {
                "csv_path": "data/india/india_regions.csv",
                "rainfall_multiplier": 1.25,
                "target_state": "Kerala",
            },
        },
        "forecast": {
            "method": "POST",
            "url": "http://127.0.0.1:8000/forecast",
            "json": {
                "windows_csv_path": "data/india/rainfall_windows.csv",
            },
        },
        "analytics_summary": {
            "method": "POST",
            "url": "http://127.0.0.1:8000/analytics/summary",
            "json": {
                "csv_path": "data/india/india_regions.csv",
            },
        },
        "graph_infer": {
            "method": "POST",
            "url": "http://127.0.0.1:8000/graph/infer",
            "json": {
                "csv_path": "data/india/india_graph_training.csv",
                "artifact_path": "artifacts/graph_gnn.pt",
            },
        },
        "config_status": {
            "method": "GET",
            "url": "http://127.0.0.1:8000/config/status",
        },
    }
    print(json.dumps(examples, indent=2))


if __name__ == "__main__":
    main()
