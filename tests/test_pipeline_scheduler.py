import sys
from pathlib import Path

import run_all
from analysis.GenerateResultsFigures import parse_args as parse_results_figure_args
from analysis.GenerateResultsFigures import merge_text_fragments


def _marker_command(order_path: Path, label: str, sleep_seconds: float = 0.0) -> tuple[str, ...]:
    code = (
        "from pathlib import Path; "
        "import time; "
        f"path = Path({str(order_path)!r}); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        f"path.open('a').write({label!r} + '\\n'); "
        f"time.sleep({sleep_seconds})"
    )
    return (sys.executable, "-c", code)


class _DummyScheduler:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.selected_completed = set()

    def run(self) -> int:
        return 0

    def abort(self) -> None:
        return None


def test_scheduler_prioritizes_ready_roots_over_blocked_children(tmp_path: Path) -> None:
    order_path = tmp_path / "order.log"
    jobs = [
        run_all.PipelineJob(
            job_id="root_a",
            deps=(),
            command=_marker_command(order_path, "A", 0.2),
            phase="data",
            resource_class="local_cpu",
            order=0,
        ),
        run_all.PipelineJob(
            job_id="child_c",
            deps=("root_a",),
            command=_marker_command(order_path, "C", 0.0),
            phase="data",
            resource_class="local_cpu",
            order=1,
        ),
        run_all.PipelineJob(
            job_id="root_b",
            deps=(),
            command=_marker_command(order_path, "B", 0.0),
            phase="data",
            resource_class="local_cpu",
            order=2,
        ),
    ]

    scheduler = run_all.PipelineScheduler(
        jobs=jobs,
        resource_profile={"local_cpu": 2},
        precompleted=set(),
        resumed_completed=set(),
        repo_root=Path.cwd(),
        logs_dir=tmp_path / "logs",
        lock_dir=tmp_path / "locks",
        resume=False,
    )

    assert scheduler.run() == 0

    execution_order = order_path.read_text().strip().splitlines()
    assert set(execution_order[:2]) == {"A", "B"}
    assert execution_order[-1] == "C"


def test_scheduler_prefers_unblocked_dependent_jobs_over_independent_heavy_roots(
    tmp_path: Path,
) -> None:
    order_path = tmp_path / "order-dependent-first.log"
    jobs = [
        run_all.PipelineJob(
            job_id="train_30b",
            deps=(),
            command=_marker_command(order_path, "train_30b", 0.2),
            phase="experiment",
            resource_class="tinker_api",
            resource_units=2,
            order=0,
        ),
        run_all.PipelineJob(
            job_id="train_235b",
            deps=(),
            command=_marker_command(order_path, "train_235b", 0.0),
            phase="experiment",
            resource_class="tinker_api",
            resource_units=2,
            order=1,
        ),
        run_all.PipelineJob(
            job_id="nudge_30b",
            deps=("train_30b",),
            command=_marker_command(order_path, "nudge_30b", 0.0),
            phase="experiment",
            resource_class="tinker_api",
            resource_units=1,
            order=2,
        ),
    ]

    scheduler = run_all.PipelineScheduler(
        jobs=jobs,
        resource_profile={"tinker_api": 2},
        precompleted=set(),
        resumed_completed=set(),
        repo_root=Path.cwd(),
        logs_dir=tmp_path / "logs",
        lock_dir=tmp_path / "locks",
        resume=False,
    )

    assert scheduler.run() == 0

    execution_order = order_path.read_text().strip().splitlines()
    assert execution_order == ["train_30b", "nudge_30b", "train_235b"]


def test_training_appendix_depends_only_on_lora_data() -> None:
    jobs = run_all.build_pipeline_jobs(python_cmd=sys.executable)
    training_appendix = next(job for job in jobs if job.job_id == "training_appendix")
    assert training_appendix.deps == ("lora_data",)


def test_skip_lora_training_marks_dependencies_precompleted() -> None:
    jobs, precompleted, _ = run_all.select_pipeline_jobs(
        machine_tag="M4",
        phase="all",
        skip_lora_training=True,
        python_cmd=sys.executable,
    )
    job_ids = {job.job_id for job in jobs}

    assert "lora_train_30b" not in job_ids
    assert "lora_train_30b_seed123" not in job_ids
    assert "lora_train_235b" not in job_ids
    assert "lora_train_235b_seed123" not in job_ids
    assert "nudge_text_30b" in job_ids
    assert {
        "lora_train_30b",
        "lora_train_30b_seed123",
        "lora_train_235b",
        "lora_train_235b_seed123",
    } <= precompleted


def test_model_filter_excludes_partial_canonical_merge_jobs() -> None:
    jobs, _, _ = run_all.select_pipeline_jobs(
        machine_tag="M4",
        phase="all",
        model_size="30b",
        python_cmd=sys.executable,
    )
    job_ids = {job.job_id for job in jobs}

    assert "merge_results_macros" not in job_ids
    assert "merge_nudge_appendix" not in job_ids
    assert "merge_similarity_appendix" not in job_ids
    assert "merge_fingerprint_appendix" not in job_ids
    assert "merge_embedding_appendix" not in job_ids
    assert "training_appendix" in job_ids
    assert "fingerprint_appendix_30b" in job_ids
    assert "fingerprint_appendix_235b" not in job_ids
    assert "embedding_appendix_30b" in job_ids
    assert "embedding_appendix_235b" not in job_ids
    assert "fp_macros_30b" in job_ids
    assert "fp_macros_235b" not in job_ids


def test_gradient_jobs_depend_on_seed_control_training() -> None:
    jobs = {job.job_id: job for job in run_all.build_pipeline_jobs(python_cmd=sys.executable)}

    assert jobs["gradient_30b"].deps == ("lora_train_30b", "lora_train_30b_seed123")
    assert jobs["gradient_235b"].deps == ("lora_train_235b", "lora_train_235b_seed123")


def test_fp_macro_jobs_depend_on_style_mismatch_fingerprint_jobs() -> None:
    jobs = {job.job_id: job for job in run_all.build_pipeline_jobs(python_cmd=sys.executable)}

    assert "fp_hds_text_30b_style" in jobs["fp_macros_30b"].deps
    assert "fp_hds_image_30b_style" in jobs["fp_macros_30b"].deps
    assert "fp_hds_text_235b_style" in jobs["fp_macros_235b"].deps
    assert "fp_hds_image_235b_style" in jobs["fp_macros_235b"].deps


def test_fingerprint_paper_jobs_pin_probe_dataset_to_hdsv2() -> None:
    jobs = {job.job_id: job for job in run_all.build_pipeline_jobs(python_cmd=sys.executable)}

    for job_id in (
        "fingerprint_figs_30b",
        "fp_macros_30b",
        "fp_macros_235b",
        "fingerprint_appendix_30b",
        "fingerprint_appendix_235b",
    ):
        command = jobs[job_id].command
        assert "--probe-hds-dataset" in command
        probe_index = command.index("--probe-hds-dataset")
        assert command[probe_index + 1] == "HDSv2"


def test_generate_results_figures_cli_defaults_probe_dataset_to_hdsv2(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["GenerateResultsFigures.py"])
    args = parse_results_figure_args()
    assert args.probe_hds_dataset == "HDSv2"


def test_generate_results_figures_cli_accepts_embedding_appendix_output_type(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["GenerateResultsFigures.py", "--output-type", "embedding-appendix"])
    args = parse_results_figure_args()
    assert args.output_type == "embedding-appendix"


def test_embedding_appendix_jobs_depend_on_image_fingerprint_runs() -> None:
    jobs = {job.job_id: job for job in run_all.build_pipeline_jobs(python_cmd=sys.executable)}

    assert jobs["embedding_appendix_30b"].deps == ("fp_hds_image_30b",)
    assert jobs["embedding_appendix_235b"].deps == ("fp_hds_image_235b",)
    assert jobs["merge_embedding_appendix"].deps == (
        "embedding_appendix_30b",
        "embedding_appendix_235b",
    )


def test_config_exposes_complexity_band_defaults() -> None:
    config = run_all.get_config()

    assert config["data_generation"]["multimodal_complexity_min"] == 10
    assert config["data_generation"]["multimodal_complexity_max"] == 324
    assert config["data_generation"]["hds_complexity_min"] == 12
    assert config["data_generation"]["hds_complexity_max"] == 324


def test_data_jobs_pass_explicit_complexity_flags() -> None:
    jobs = {job.job_id: job for job in run_all.build_pipeline_jobs(python_cmd=sys.executable)}

    assert jobs["hds_gen"].command[-4:] == ("--complexity-min", "12", "--complexity-max", "324")
    assert jobs["multimodal_grid"].command[-4:] == ("--complexity-min", "10", "--complexity-max", "324")
    assert jobs["math_text"].command[-4:] == ("--complexity-min", "10", "--complexity-max", "324")
    assert jobs["math_images"].command[-4:] == ("--complexity-min", "10", "--complexity-max", "324")
    assert jobs["math_audio"].command[-4:] == ("--complexity-min", "10", "--complexity-max", "324")


def test_merge_text_fragments_preserves_input_order(tmp_path: Path) -> None:
    fragment_30b = tmp_path / "macros.30b.tex"
    fragment_235b = tmp_path / "macros.235b.tex"
    output_path = tmp_path / "results_macros.tex"

    fragment_30b.write_text("% 30B\n\\newcommand{\\ThirtyBOnly}{1}\n")
    fragment_235b.write_text("% 235B\n\\newcommand{\\TwoThirtyFiveBOnly}{1}\n")

    merge_text_fragments([fragment_30b, fragment_235b], output_path)

    merged = output_path.read_text()
    assert merged.index("\\ThirtyBOnly") < merged.index("\\TwoThirtyFiveBOnly")


def test_scheduler_uses_devnull_stdin_for_child_jobs(monkeypatch, tmp_path: Path) -> None:
    popen_kwargs = {}

    class DummyProcess:
        def __init__(self) -> None:
            self.stdout = []
            self.stderr = []

        def poll(self) -> int:
            return 0

    def fake_popen(*args, **kwargs):
        popen_kwargs.update(kwargs)
        return DummyProcess()

    monkeypatch.setattr(run_all.subprocess, "Popen", fake_popen)

    scheduler = run_all.PipelineScheduler(
        jobs=[
            run_all.PipelineJob(
                job_id="one",
                deps=(),
                command=(sys.executable, "-c", "print('ok')"),
                phase="data",
                resource_class="local_cpu",
                order=0,
            )
        ],
        resource_profile={"local_cpu": 1},
        precompleted=set(),
        resumed_completed=set(),
        repo_root=Path.cwd(),
        logs_dir=tmp_path / "logs",
        lock_dir=tmp_path / "locks",
        resume=False,
    )

    monkeypatch.setattr(
        scheduler,
        "_stream_pipe",
        lambda *args, **kwargs: None,
    )

    assert scheduler.run() == 0
    assert popen_kwargs["stdin"] is run_all.subprocess.DEVNULL
    assert popen_kwargs["env"]["PYTHONUNBUFFERED"] == "1"


def test_scheduler_preserves_explicit_pythonunbuffered_override(monkeypatch, tmp_path: Path) -> None:
    popen_kwargs = {}

    class DummyProcess:
        def __init__(self) -> None:
            self.stdout = []
            self.stderr = []

        def poll(self) -> int:
            return 0

    def fake_popen(*args, **kwargs):
        popen_kwargs.update(kwargs)
        return DummyProcess()

    monkeypatch.setattr(run_all.subprocess, "Popen", fake_popen)

    scheduler = run_all.PipelineScheduler(
        jobs=[
            run_all.PipelineJob(
                job_id="one",
                deps=(),
                command=(sys.executable, "-c", "print('ok')"),
                phase="data",
                resource_class="local_cpu",
                order=0,
            )
        ],
        resource_profile={"local_cpu": 1},
        precompleted=set(),
        resumed_completed=set(),
        repo_root=Path.cwd(),
        logs_dir=tmp_path / "logs",
        lock_dir=tmp_path / "locks",
        resume=False,
        env={"PYTHONUNBUFFERED": "0"},
    )

    monkeypatch.setattr(
        scheduler,
        "_stream_pipe",
        lambda *args, **kwargs: None,
    )

    assert scheduler.run() == 0
    assert popen_kwargs["env"]["PYTHONUNBUFFERED"] == "0"


def test_run_pipeline_skips_tinker_preflight_for_data_phase(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        run_all,
        "select_pipeline_jobs",
        lambda **kwargs: ([], set(), {"local_cpu": 1}),
    )
    monkeypatch.setattr(run_all, "PipelineScheduler", _DummyScheduler)

    def fail_if_called() -> str:
        raise AssertionError("validate_tinker_runtime should not be called for data phase")

    monkeypatch.setattr(run_all, "validate_tinker_runtime", fail_if_called)

    exit_code = run_all.run_pipeline(
        machine_tag="M4",
        phase="data",
        dry_run=False,
        skip_lora_training=False,
        model_size=None,
        resume=False,
        logs_dir=tmp_path / "logs",
        lock_dir=tmp_path / "locks",
    )

    assert exit_code == 0


def test_run_pipeline_requires_tinker_api_key_before_preflight(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(
        run_all,
        "select_pipeline_jobs",
        lambda **kwargs: ([], set(), {"local_cpu": 1}),
    )
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.setattr(run_all, "PipelineScheduler", _DummyScheduler)

    def fail_if_called() -> str:
        raise AssertionError("validate_tinker_runtime should not be called without API key")

    monkeypatch.setattr(run_all, "validate_tinker_runtime", fail_if_called)

    exit_code = run_all.run_pipeline(
        machine_tag="M4",
        phase="experiment",
        dry_run=False,
        skip_lora_training=False,
        model_size=None,
        resume=False,
        logs_dir=tmp_path / "logs",
        lock_dir=tmp_path / "locks",
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "TINKER_API_KEY not set" in captured.err


def test_run_pipeline_fails_fast_when_tinker_preflight_fails(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(
        run_all,
        "select_pipeline_jobs",
        lambda **kwargs: ([], set(), {"local_cpu": 1}),
    )
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    monkeypatch.setattr(
        run_all,
        "validate_tinker_runtime",
        lambda: "ERROR: Tinker runtime preflight failed for interpreter /tmp/python.",
    )

    class _FailIfConstructed:
        def __init__(self, **kwargs) -> None:
            raise AssertionError("PipelineScheduler should not be constructed on preflight failure")

    monkeypatch.setattr(run_all, "PipelineScheduler", _FailIfConstructed)

    exit_code = run_all.run_pipeline(
        machine_tag="M4",
        phase="all",
        dry_run=False,
        skip_lora_training=False,
        model_size=None,
        resume=False,
        logs_dir=tmp_path / "logs",
        lock_dir=tmp_path / "locks",
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Tinker runtime preflight failed" in captured.err


def test_run_pipeline_runs_when_tinker_preflight_succeeds(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        run_all,
        "select_pipeline_jobs",
        lambda **kwargs: ([], set(), {"local_cpu": 1}),
    )
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    monkeypatch.setattr(run_all, "validate_tinker_runtime", lambda: None)
    monkeypatch.setattr(run_all, "PipelineScheduler", _DummyScheduler)

    exit_code = run_all.run_pipeline(
        machine_tag="M4",
        phase="experiment",
        dry_run=False,
        skip_lora_training=False,
        model_size=None,
        resume=False,
        logs_dir=tmp_path / "logs",
        lock_dir=tmp_path / "locks",
    )

    assert exit_code == 0
