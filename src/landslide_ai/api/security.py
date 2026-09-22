from __future__ import annotations

from fastapi import Header, HTTPException

from landslide_ai.config import AppConfig


def require_api_key(config: AppConfig, api_key_header_name: str = "X-API-Key"):
    async def dependency(x_api_key: str | None = Header(default=None, alias=api_key_header_name)) -> None:
        if not config.api_auth_enabled:
            return
        if x_api_key != config.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return dependency
