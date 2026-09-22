from importlib.util import find_spec


def main() -> None:
    if find_spec("fastapi") is None or find_spec("uvicorn") is None:
        print("FastAPI dependencies are not installed. Install `fastapi` and `uvicorn` first.")
        return

    import os

    import uvicorn
    from landslide_ai.config import load_config

    config = load_config(".")
    port = config.api_port
    platform_port = os.getenv("PORT")
    if platform_port:
        port = int(platform_port)
    uvicorn.run(
        "landslide_ai.api.app:app",
        host=config.api_host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
