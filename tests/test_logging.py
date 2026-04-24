import builtins

from core.Logging import tprint


def test_tprint_flushes_by_default(monkeypatch) -> None:
    print_calls = []

    def fake_print(*args, **kwargs) -> None:
        print_calls.append((args, kwargs))

    monkeypatch.setattr(builtins, "print", fake_print)

    tprint("hello")

    assert len(print_calls) == 1
    args, kwargs = print_calls[0]
    assert args[1] == "hello"
    assert kwargs["flush"] is True


def test_tprint_preserves_explicit_flush_override(monkeypatch) -> None:
    print_calls = []

    def fake_print(*args, **kwargs) -> None:
        print_calls.append((args, kwargs))

    monkeypatch.setattr(builtins, "print", fake_print)

    tprint("hello", flush=False)

    assert len(print_calls) == 1
    _, kwargs = print_calls[0]
    assert kwargs["flush"] is False
