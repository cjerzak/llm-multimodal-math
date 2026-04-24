#!/usr/bin/env python3
"""
Shared Tinker startup configuration and ServiceClient bootstrap helpers.
"""

from __future__ import annotations

import os
import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional


DEFAULT_TINKER_STARTUP_DEADLINE_SECONDS = 250.0
DEFAULT_TINKER_STARTUP_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_TINKER_STARTUP_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_TINKER_STARTUP_MAX_RETRIES = 3


class TinkerStartupTimeoutError(TimeoutError):
    """Raised when Tinker bootstrap exceeds the configured wall-clock deadline."""


@dataclass(frozen=True)
class TinkerStartupConfig:
    deadline_seconds: float = DEFAULT_TINKER_STARTUP_DEADLINE_SECONDS
    request_timeout_seconds: float = DEFAULT_TINKER_STARTUP_REQUEST_TIMEOUT_SECONDS
    connect_timeout_seconds: float = DEFAULT_TINKER_STARTUP_CONNECT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_TINKER_STARTUP_MAX_RETRIES


def _read_float_env(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
) -> float:
    raw_value = env.get(name)
    if raw_value is None or raw_value == "":
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw_value!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {raw_value!r}")
    return value


def _read_int_env(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    raw_value = env.get(name)
    if raw_value is None or raw_value == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {raw_value!r}")
    return value


def load_tinker_startup_config(env: Optional[Mapping[str, str]] = None) -> TinkerStartupConfig:
    """Load startup configuration from environment variables."""
    env = env or os.environ
    return TinkerStartupConfig(
        deadline_seconds=_read_float_env(
            env,
            "TINKER_STARTUP_DEADLINE_SECONDS",
            DEFAULT_TINKER_STARTUP_DEADLINE_SECONDS,
            minimum=0.0,
        ),
        request_timeout_seconds=_read_float_env(
            env,
            "TINKER_STARTUP_REQUEST_TIMEOUT_SECONDS",
            DEFAULT_TINKER_STARTUP_REQUEST_TIMEOUT_SECONDS,
            minimum=0.0,
        ),
        connect_timeout_seconds=_read_float_env(
            env,
            "TINKER_STARTUP_CONNECT_TIMEOUT_SECONDS",
            DEFAULT_TINKER_STARTUP_CONNECT_TIMEOUT_SECONDS,
            minimum=0.0,
        ),
        max_retries=_read_int_env(
            env,
            "TINKER_STARTUP_MAX_RETRIES",
            DEFAULT_TINKER_STARTUP_MAX_RETRIES,
            minimum=0,
        ),
    )


def _format_seconds(value: float) -> str:
    text = f"{value:g}"
    return f"{text}s"


def format_tinker_startup_config(config: TinkerStartupConfig) -> str:
    """Return a compact human-readable startup settings string."""
    return (
        f"deadline={_format_seconds(config.deadline_seconds)}, "
        f"request_timeout={_format_seconds(config.request_timeout_seconds)}, "
        f"connect_timeout={_format_seconds(config.connect_timeout_seconds)}, "
        f"max_retries={config.max_retries}"
    )


def _supports_signal_deadline() -> bool:
    return (
        os.name == "posix"
        and threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and hasattr(signal, "getitimer")
    )


@contextmanager
def _startup_deadline_context(seconds: float) -> Iterator[None]:
    """Apply a wall-clock deadline around Tinker bootstrap on POSIX main threads."""
    if seconds <= 0 or not _supports_signal_deadline():
        yield
        return

    def _handle_timeout(_signum: int, _frame: Any) -> None:
        raise TinkerStartupTimeoutError(
            f"Tinker startup exceeded the {seconds:g}s wall-clock deadline"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def build_service_client_kwargs(
    tinker_module: Any,
    *,
    api_key: Optional[str] = None,
    config: Optional[TinkerStartupConfig] = None,
) -> dict[str, Any]:
    """Build the supported kwargs for tinker.ServiceClient startup."""
    config = config or load_tinker_startup_config()
    kwargs: dict[str, Any] = {
        "timeout": tinker_module.Timeout(
            timeout=config.request_timeout_seconds,
            connect=config.connect_timeout_seconds,
        ),
        "max_retries": config.max_retries,
    }
    if api_key is not None:
        kwargs["api_key"] = api_key
    return kwargs


def create_tinker_service_client(
    *,
    tinker_module: Any,
    api_key: Optional[str] = None,
    config: Optional[TinkerStartupConfig] = None,
) -> Any:
    """Create a ServiceClient with bounded startup settings."""
    config = config or load_tinker_startup_config()
    kwargs = build_service_client_kwargs(
        tinker_module,
        api_key=api_key,
        config=config,
    )
    with _startup_deadline_context(config.deadline_seconds):
        return tinker_module.ServiceClient(**kwargs)
