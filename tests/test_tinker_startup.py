import os
import signal
import subprocess
import sys
import textwrap
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_all
from core.TinkerStartup import (
    TinkerStartupConfig,
    TinkerStartupTimeoutError,
    create_tinker_service_client,
)

tinker_client_module = importlib.import_module("core.TinkerClient")


class _FakeTimeout:
    def __init__(self, *, timeout, connect):
        self.timeout = timeout
        self.connect = connect


class _FakeTinkerModule:
    Timeout = _FakeTimeout

    def __init__(self) -> None:
        self.calls = []

    def ServiceClient(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(kwargs=kwargs)


def test_create_tinker_service_client_forwards_timeout_settings() -> None:
    fake_tinker = _FakeTinkerModule()
    startup_config = TinkerStartupConfig(
        deadline_seconds=250,
        request_timeout_seconds=120,
        connect_timeout_seconds=5,
        max_retries=3,
    )

    client = create_tinker_service_client(
        tinker_module=fake_tinker,
        api_key="test-key",
        config=startup_config,
    )

    kwargs = fake_tinker.calls[0]
    assert client.kwargs is kwargs
    assert kwargs["api_key"] == "test-key"
    assert kwargs["max_retries"] == 3
    assert kwargs["timeout"].timeout == 120
    assert kwargs["timeout"].connect == 5


def test_run_startup_phase_logs_start_and_done(monkeypatch) -> None:
    logs = []
    monkeypatch.setattr(tinker_client_module, "tprint", logs.append)

    result = tinker_client_module._run_startup_phase(
        phase_name="init_api",
        model_name="fake-model",
        startup_config=TinkerStartupConfig(),
        verbose=True,
        action=lambda: "ok",
    )

    assert result == "ok"
    assert logs[0] == "  init_api: start"
    assert logs[1].startswith("  init_api: done in ")


def test_run_startup_phase_wraps_failure_with_context(monkeypatch) -> None:
    logs = []
    monkeypatch.setattr(tinker_client_module, "tprint", logs.append)

    def _boom() -> None:
        raise TimeoutError("stalled")

    with pytest.raises(RuntimeError) as excinfo:
        tinker_client_module._run_startup_phase(
            phase_name="init_api",
            model_name="fake-model",
            startup_config=TinkerStartupConfig(),
            verbose=True,
            action=_boom,
        )

    message = str(excinfo.value)
    assert "startup failed during `init_api`" in message
    assert "`fake-model`" in message
    assert f"`{sys.executable}`" in message
    assert "deadline=250s" in message
    assert "request_timeout=120s" in message
    assert "connect_timeout=5s" in message
    assert "max_retries=3" in message
    assert "TimeoutError: stalled" in message
    assert logs[0] == "  init_api: start"
    assert logs[1].startswith("  init_api: failed after ")


def test_validate_tinker_runtime_reports_startup_settings(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tinker", SimpleNamespace())
    monkeypatch.setattr(
        run_all,
        "create_tinker_service_client",
        lambda **kwargs: (_ for _ in ()).throw(TimeoutError("stalled")),
    )
    monkeypatch.setattr(
        run_all,
        "load_tinker_startup_config",
        lambda: TinkerStartupConfig(),
    )

    message = run_all.validate_tinker_runtime()

    assert message is not None
    assert "during `init_api`" in message
    assert "Startup settings: deadline=250s, request_timeout=120s, connect_timeout=5s, max_retries=3" in message
    assert "TimeoutError: stalled" in message


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(signal, "setitimer"),
    reason="requires POSIX signal timers",
)
def test_create_tinker_service_client_enforces_outer_deadline_in_subprocess() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "Scripts"
    code = textwrap.dedent(
        f"""
        import sys
        import time

        sys.path.insert(0, {str(repo_root)!r})
        sys.path.insert(0, {str(scripts_dir)!r})

        from core.TinkerStartup import (
            TinkerStartupConfig,
            TinkerStartupTimeoutError,
            create_tinker_service_client,
        )

        class FakeTinker:
            class Timeout:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

            @staticmethod
            def ServiceClient(**kwargs):
                time.sleep(1.0)

        try:
            create_tinker_service_client(
                tinker_module=FakeTinker,
                config=TinkerStartupConfig(
                    deadline_seconds=0.2,
                    request_timeout_seconds=1,
                    connect_timeout_seconds=0.1,
                    max_retries=3,
                ),
            )
        except TinkerStartupTimeoutError as exc:
            print(type(exc).__name__)
            sys.exit(0)

        sys.exit(1)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=repo_root,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "TinkerStartupTimeoutError" in result.stdout
