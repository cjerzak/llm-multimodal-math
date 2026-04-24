from pathlib import Path

import pytest

from analysis.GenerateResultsFigures import (
    compute_embedding_support_mass_from_detail,
    compute_probe_dataset_summary,
    compute_support_mass_from_detail,
    compute_table3_stats_from_jsonl,
    generate_embedding_results_appendix,
    generate_jsonl_alt_macros,
    generate_latex_macros,
    generate_nudge_examples,
    generate_similarity_matrix,
    generate_template_variability_appendix,
    merge_embedding_results_appendix_fragments,
    infer_loss_best_heuristic_from_detail,
    load_analysis,
    merge_nudge_appendix_fragments,
    merge_similarity_appendix_fragments,
    merge_template_variability_appendix_fragments,
    merge_text_fragments,
)
from core.TinkerClient import (
    get_effective_heuristic_template_metadata,
    set_heuristic_template_mode,
    set_heuristic_template_profile,
    validate_active_heuristic_templates,
)
from experiments.BaselineFingerprint import (
    LOSS_DETECTION_SEMANTICS,
    FingerprintingResult,
    _initialize_embedding_classifier,
    _format_detail_record,
    analyze_results,
    load_hds as load_fingerprint_hds,
    resolve_images_dir,
    select_rows_for_split as select_fingerprint_rows_for_split,
)
from experiments.ContrastiveStepProbe import (
    load_hds as load_contrastive_hds,
    select_rows_for_split as select_contrastive_rows_for_split,
)


def test_detail_record_preserves_loss_detection_and_diagnostics() -> None:
    set_heuristic_template_mode("multi", seed=7)
    set_heuristic_template_profile("style_mismatch")

    result = FingerprintingResult(
        hds_id="hds_001",
        a=12,
        b=34,
        product=408,
        target_heuristic="DD",
        detected_heuristic="DD",
        detection_confidence=0.42,
        model_answer=406,
        is_correct=False,
        error_delta=-2,
        error_heuristic="rounding_compensation",
        error_confidence=0.8,
        perplexity_losses={"DD": 1.0, "OT": 1.8, "RC": 2.1},
        trace="12 x 34",
        trace_heuristic="ones_then_tens",
        trace_confidence=0.7,
        embedding_heuristic="DD",
        embedding_confidence=0.88,
        embedding_margin=0.64,
        embedding_support_mass={"DD": 0.88, "OT": 0.04, "RC": 0.03, "STYLE": 0.05},
        embedding_model="fake-local-embedder",
        embedding_resolved=True,
        embedding_resolution_status="ok",
        neutral_loss=2.5,
        delta_losses={"DD": -1.5, "OT": -0.7, "RC": -0.4},
        per_template_losses={"DD_0": {"prompt": "test", "loss": 1.0}},
        extraction_confidence=0.9,
        extraction_strategy="answer_marker",
        is_truncated=False,
        is_contaminated=False,
    )

    record = _format_detail_record(result)

    assert record["detection_semantics"] == LOSS_DETECTION_SEMANTICS
    assert record["template_profile"] == "style_mismatch"
    assert record["template_profile_kind"] == "style_only"
    assert record["template_mode"] == "multi"
    assert record["perplexity"]["loss_best_heuristic"] == "DD"
    assert record["perplexity"]["loss_best_confidence"] == pytest.approx(0.42)
    assert "detected_heuristic" not in record["perplexity"]
    assert record["trace_analysis"]["trace_heuristic"] == "ones_then_tens"
    assert record["error_analysis"]["error_heuristic"] == "rounding_compensation"
    assert record["embedding_analysis"]["embedding_heuristic"] == "DD"
    assert record["embedding_analysis"]["model"] == "fake-local-embedder"


def test_embedding_classifier_preflight_fails_before_generation(monkeypatch, tmp_path: Path) -> None:
    class _BrokenClassifier:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def warmup(self):
            raise RuntimeError("cache miss")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        "experiments.BaselineFingerprint.PrototypeEmbeddingClassifier",
        _BrokenClassifier,
    )

    with pytest.raises(RuntimeError, match="Failed to initialize the trace embedding detector before generation"):
        _initialize_embedding_classifier(
            "Qwen/Qwen3-Embedding-0.6B",
            cache_dir=tmp_path / "cache",
            batch_size=4,
            prototype_sample_cap=8,
        )


def test_table3_stats_recompute_argmin_from_aggregated_losses() -> None:
    records = [
        {
            "target_heuristic": "DD",
            "generation": {"is_correct": True},
            "perplexity": {
                "aggregated": {"DD": 1.0, "OT": 2.0, "RC": 3.0},
                "neutral_loss": 4.0,
                "detected_heuristic": "RC",
            },
        },
        {
            "target_heuristic": "OT",
            "generation": {"is_correct": False},
            "perplexity": {
                "aggregated": {"DD": 3.0, "OT": 1.0, "RC": 2.0},
                "neutral_loss": 4.0,
                "detected_heuristic": "DD",
            },
        },
    ]

    stats = compute_table3_stats_from_jsonl(records)

    assert stats is not None
    assert stats["detection_rates"]["DD"]["rate"] == pytest.approx(1.0)
    assert stats["detection_rates"]["OT"]["rate"] == pytest.approx(1.0)
    assert stats["delta_loss"]["DD"] == pytest.approx(-2.0)
    assert stats["resolved_probe_count"] == 2
    assert stats["resolved_probe_rate"] == pytest.approx(1.0)
    assert stats["soft_target_stats"]["DD"]["coverage_rate"] == pytest.approx(1.0)
    assert stats["soft_target_stats"]["OT"]["coverage_rate"] == pytest.approx(1.0)
    assert stats["soft_target_stats"]["DD"]["target_support"] == pytest.approx(
        compute_support_mass_from_detail(records[0])["DD"]
    )
    assert stats["soft_target_stats"]["OT"]["target_support"] == pytest.approx(
        compute_support_mass_from_detail(records[1])["OT"]
    )


def test_table3_stats_excludes_unresolved_rows_from_target_support() -> None:
    records = [
        {
            "target_heuristic": "DD",
            "generation": {"is_correct": True},
            "perplexity": {
                "aggregated": {"DD": 0.5, "OT": 1.5, "RC": 2.5},
                "neutral_loss": 3.5,
                "probe_resolved": True,
            },
        },
        {
            "target_heuristic": "DD",
            "generation": {"is_correct": False},
            "perplexity": {
                "aggregated": {"DD": float("inf"), "OT": float("inf"), "RC": float("inf")},
                "neutral_loss": float("inf"),
                "probe_resolved": False,
            },
        },
    ]

    stats = compute_table3_stats_from_jsonl(records)

    assert stats is not None
    assert stats["resolved_probe_count"] == 1
    assert stats["resolved_probe_rate"] == pytest.approx(0.5)
    dd_stats = stats["soft_target_stats"]["DD"]
    assert dd_stats["total_n"] == pytest.approx(2.0)
    assert dd_stats["resolved_n"] == pytest.approx(1.0)
    assert dd_stats["coverage_rate"] == pytest.approx(0.5)
    assert dd_stats["target_support"] == pytest.approx(
        compute_support_mass_from_detail(records[0])["DD"]
    )


def test_table3_stats_support_mass_tracks_neutral_when_neutral_is_best() -> None:
    record = {
        "target_heuristic": "RC",
        "generation": {"is_correct": True},
        "perplexity": {
            "aggregated": {"DD": 2.0, "OT": 2.5, "RC": 3.0},
            "neutral_loss": 1.0,
        },
    }

    support_mass = compute_support_mass_from_detail(record)

    assert support_mass is not None
    assert support_mass["NEUTRAL"] > support_mass["DD"]
    assert support_mass["NEUTRAL"] > support_mass["OT"]
    assert support_mass["NEUTRAL"] > support_mass["RC"]

    stats = compute_table3_stats_from_jsonl([record])

    assert stats is not None
    assert stats["soft_target_stats"]["RC"]["target_support"] == pytest.approx(support_mass["RC"])
    assert stats["soft_target_stats"]["RC"]["target_support"] < 0.5


def test_table3_stats_include_embedding_support_with_style_competitor() -> None:
    record = {
        "target_heuristic": "DD",
        "generation": {"is_correct": True},
        "perplexity": {
            "aggregated": {"DD": 1.0, "OT": 2.0, "RC": 3.0},
            "neutral_loss": 4.0,
        },
        "embedding_analysis": {
            "embedding_heuristic": "DD",
            "resolved": True,
            "support_mass": {"DD": 0.7, "OT": 0.1, "RC": 0.1, "STYLE": 0.1},
        },
    }

    stats = compute_table3_stats_from_jsonl([record])

    assert stats is not None
    assert compute_embedding_support_mass_from_detail(record)["STYLE"] == pytest.approx(0.1)
    assert stats["resolved_embedding_count"] == 1
    assert stats["resolved_embedding_rate"] == pytest.approx(1.0)
    dd_stats = stats["embedding_soft_target_stats"]["DD"]
    assert dd_stats["coverage_rate"] == pytest.approx(1.0)
    assert dd_stats["target_support"] == pytest.approx(0.7)
    assert dd_stats["detection_rate"] == pytest.approx(1.0)


def test_generate_jsonl_alt_macros_emits_coverage_and_target_support_macros() -> None:
    text_records = [
        {
            "target_heuristic": "DD",
            "generation": {"is_correct": True},
            "perplexity": {
                "aggregated": {"DD": 0.5, "OT": 1.5, "RC": 2.5},
                "neutral_loss": 3.5,
                "probe_resolved": True,
            },
            "embedding_analysis": {
                "embedding_heuristic": "DD",
                "resolved": True,
                "support_mass": {"DD": 0.7, "OT": 0.1, "RC": 0.1, "STYLE": 0.1},
            },
        }
    ]
    image_records = [
        {
            "target_heuristic": "DD",
            "generation": {"is_correct": True},
            "perplexity": {
                "aggregated": {"DD": float("inf"), "OT": float("inf"), "RC": float("inf")},
                "neutral_loss": float("inf"),
                "probe_resolved": False,
            },
            "embedding_analysis": {
                "resolved": False,
            },
        }
    ]

    macros = generate_jsonl_alt_macros(text_records, image_records, "ThirtyB")

    assert r"\newcommand{\HDSTestResolvedCoverageAltThirtyB}{100.0\%}" in macros
    assert r"\newcommand{\HDSTestDDCoverageAltThirtyB}{100.0\%}" in macros
    assert r"\newcommand{\HDSTestDDTargetSupportAltThirtyB}{" in macros
    assert r"\newcommand{\HDSTestResolvedCoverageImageAltThirtyB}{0.0\%}" in macros
    assert r"\newcommand{\HDSTestDDCoverageImageAltThirtyB}{0.0\%}" in macros
    assert r"\newcommand{\HDSTestDDTargetSupportImageAltThirtyB}{0.0\%}" in macros
    assert r"\newcommand{\HDSTestResolvedTraceEmbedCoverageAltThirtyB}{100.0\%}" in macros
    assert r"\newcommand{\HDSTestDDTraceEmbedTargetSupportAltThirtyB}{70.0\%}" in macros
    assert r"\newcommand{\HDSTestResolvedTraceEmbedCoverageImageAltThirtyB}{0.0\%}" in macros


def test_compute_probe_dataset_summary_matches_hds_test_split_mean_complexity() -> None:
    summary = compute_probe_dataset_summary("HDS")

    assert summary["count"] == 144
    assert summary["mean_c"] == pytest.approx(28.5625)
    assert summary["mean_c_display"] == "28.6"


def test_compute_probe_dataset_summary_filters_to_evaluated_ids(monkeypatch, tmp_path) -> None:
    dataset_path = tmp_path / "ToyProbe.csv"
    dataset_path.write_text(
        "id,a,b,product,target_heuristic,ot_score,dd_score,rc_score,category,notes,complexity_c,split\n"
        "p1,12,13,156,DD,0.1,0.8,0.2,cat,note,10,test\n"
        "p2,14,15,210,OT,0.2,0.1,0.8,cat,note,20,test\n"
        "p3,16,17,272,RC,0.2,0.2,0.7,cat,note,30,test\n"
    )

    monkeypatch.setattr(
        "analysis.GenerateResultsFigures.get_dataset_csv_path",
        lambda dataset_name: dataset_path,
    )

    summary = compute_probe_dataset_summary(
        "ToyProbe",
        result_rows=[{"hds_id": "p1"}],
        detail_rows=[{"hds_id": "p3"}],
    )

    assert summary["count"] == 2
    assert summary["mean_c"] == pytest.approx(20.0)
    assert summary["mean_c_display"] == "20.0"


def test_generate_latex_macros_emits_probe_mean_c_and_preserves_canonical_hds_counts(monkeypatch) -> None:
    monkeypatch.setattr("analysis.GenerateResultsFigures.get_multimodal_dataset_size", lambda: 0)
    monkeypatch.setattr("analysis.GenerateResultsFigures.get_hds_dataset_size", lambda: 1000)
    monkeypatch.setattr("analysis.GenerateResultsFigures.get_traps_dataset_size", lambda: 30)
    monkeypatch.setattr(
        "analysis.GenerateResultsFigures.get_hds_split_composition",
        lambda: {
            "train": {"RC": 230, "DD": 231, "OT": 240},
            "val": {"RC": 50, "DD": 52, "OT": 53},
            "test": {"RC": 48, "DD": 48, "OT": 48},
        },
    )
    monkeypatch.setattr(
        "analysis.GenerateResultsFigures.compute_probe_dataset_summary",
        lambda *args, **kwargs: {
            "dataset_name": "HDSHard",
            "count": 17,
            "mean_c": 42.125,
            "mean_c_display": "42.1",
            "dataset_path": "/tmp/HDSHard.csv",
        },
    )

    macros = generate_latex_macros(
        hds_test={"total": 17, "accuracy": 0.5, "avg_perplexity": {}, "by_target_heuristic": {}},
        hds_all=None,
        traps={},
        probe_hds_dataset_name="HDSHard",
        include_global_macros=True,
        emit_contrastive_placeholders=False,
        emit_gradient_placeholders=False,
        emit_suffix_macros=False,
    )

    assert r"\newcommand{\HDSTestCount}{144}" in macros
    assert r"\newcommand{\HDSProbeCount}{17}" in macros
    assert r"\newcommand{\HDSProbeMeanCExact}{42.1250}" in macros
    assert r"\newcommand{\HDSProbeMeanCDisplay}{42.1}" in macros
    assert r"\newcommand{\HDSProbeAccuracyMeanCLabel}{Accuracy (Mean C = 42.1)}" in macros


def test_ms_tex_uses_generated_probe_mean_c_label_macro() -> None:
    ms_tex = (Path(__file__).resolve().parents[1] / "PaperTexFolder" / "ms.tex").read_text()

    assert (
        r"\providecommand{\HDSProbeAccuracyMeanCLabel}{Accuracy (Mean C = \HDSProbeMeanCDisplay{})}"
        in ms_tex
    )
    assert ms_tex.count(r"\HDSProbeAccuracyMeanCLabel{}") == 2


def test_custom_csv_split_selection_prefers_explicit_split_column(tmp_path) -> None:
    dataset_path = tmp_path / "HDSHard.csv"
    dataset_path.write_text(
        "id,a,b,product,target_heuristic,ot_score,dd_score,rc_score,category,notes,split\n"
        "p1,12,13,156,DD,0.1,0.8,0.2,cat,note,train\n"
        "p2,14,15,210,OT,0.2,0.1,0.8,cat,note,test\n"
        "p3,16,17,272,RC,0.2,0.2,0.7,cat,note,test\n"
    )

    fingerprint_rows = load_fingerprint_hds(dataset_path)
    contrastive_rows = load_contrastive_hds(dataset_path)

    assert [row.split for row in fingerprint_rows] == ["train", "test", "test"]
    assert [row.split for row in contrastive_rows] == ["train", "test", "test"]
    assert [row.id for row in select_fingerprint_rows_for_split(fingerprint_rows, "test", "HDSHard")] == ["p2", "p3"]
    assert [row.id for row in select_contrastive_rows_for_split(contrastive_rows, "test", "HDSHard")] == ["p2", "p3"]


def test_resolve_images_dir_defaults_to_custom_dataset_images() -> None:
    default_path = resolve_images_dir("HDSHard", has_custom_csv=True)
    assert default_path is not None
    assert default_path.name == "HDSHardImages"
    assert default_path.parent.name == "SavedData"
    assert resolve_images_dir("HDSHard", images_dir_arg="/tmp/custom", has_custom_csv=True) == Path("/tmp/custom")
    default_canonical = resolve_images_dir("HDSv2", has_custom_csv=False)
    assert default_canonical is not None
    assert default_canonical.name == "HDSv2Images"


def test_detail_recompute_rejects_inconsistent_serialized_loss_best() -> None:
    record = {
        "perplexity": {
            "aggregated": {"DD": 0.8, "OT": 1.4, "RC": 1.9},
            "loss_best_heuristic": "RC",
        }
    }

    with pytest.raises(ValueError):
        infer_loss_best_heuristic_from_detail(record)


def test_generate_latex_macros_excludes_style_from_headline_nudge_totals() -> None:
    nudge_test = {
        "RC": {
            "total_problems": 10,
            "by_target_heuristic": {"RC": {"base_correct": 3, "base_known_total": 5, "total": 5}},
            "flips": {"total": 1, "improved": 0, "degraded": 1},
            "detection_flips": {"total": 10},
        },
        "DD": {
            "total_problems": 10,
            "by_target_heuristic": {"DD": {"base_correct": 3, "base_known_total": 5, "total": 5}},
            "flips": {"total": 2, "improved": 1, "degraded": 1},
            "detection_flips": {"total": 11},
        },
        "OT": {
            "total_problems": 10,
            "by_target_heuristic": {"OT": {"base_correct": 3, "base_known_total": 5, "total": 5}},
            "flips": {"total": 3, "improved": 1, "degraded": 2},
            "detection_flips": {"total": 12},
        },
        "STYLE": {
            "total_problems": 10,
            "by_target_heuristic": {"RC": {"base_correct": 3, "base_known_total": 5, "total": 5}},
            "flips": {"total": 4, "improved": 1, "degraded": 3},
            "detection_flips": {"total": 13},
        },
    }
    nudge_test_results = [
        {"hds_id": "p1", "lora": "RC", "target_heuristic": "RC", "base_correctness": "correct", "base_detected": "RC", "lora_detected": "RC"},
        {"hds_id": "p2", "lora": "DD", "target_heuristic": "DD", "base_correctness": "correct", "base_detected": "RC", "lora_detected": "DD"},
        {"hds_id": "p3", "lora": "OT", "target_heuristic": "OT", "base_correctness": "incorrect", "base_detected": "DD", "lora_detected": "OT"},
        {"hds_id": "p4", "lora": "STYLE", "target_heuristic": "RC", "base_correctness": "correct", "base_detected": "RC", "lora_detected": "DD"},
    ]
    nudge_test_details = [
        {
            "a": 12,
            "b": 12,
            "base_evaluation": {"model_answer": 144},
            "lora_evaluations": {
                "rc_lora": {"model_answer": 144, "correctness_flip_type": "unchanged"},
                "style_lora": {"model_answer": 1440, "correctness_flip_type": "degraded"},
            },
        }
    ]

    macros = generate_latex_macros(
        hds_test={},
        hds_all=None,
        traps={},
        nudge_test=nudge_test,
        nudge_test_results=nudge_test_results,
        nudge_test_details=nudge_test_details,
        include_global_macros=False,
        emit_contrastive_placeholders=False,
        emit_gradient_placeholders=False,
        emit_suffix_macros=False,
    )

    assert r"\newcommand{\NudgeLoRACount}{3}" in macros
    assert r"\newcommand{\NudgeTotalComparisons}{30}" in macros
    assert r"\newcommand{\NudgeTotalFlips}{6}" in macros
    assert r"\newcommand{\NudgeDetectionFlips}{33}" in macros
    assert r"\newcommand{\StyleControlFlips}{4}" in macros
    assert r"\newcommand{\NudgeDegradedMagnitudeSlip}{0}" in macros


def test_template_validation_tracks_requested_profile() -> None:
    set_heuristic_template_mode("multi", seed=0)
    set_heuristic_template_profile("style_mismatch")

    validation = validate_active_heuristic_templates(
        expected_profile="style_mismatch",
        expected_mode="multi",
        expected_seed=0,
    )
    balanced = get_effective_heuristic_template_metadata(
        profile="balanced",
        mode="multi",
        seed=0,
    )

    assert validation["is_valid"] is True
    assert validation["config_matches"] is True
    assert validation["template_profile"] == "style_mismatch"
    assert validation["template_profile_kind"] == "style_only"
    assert validation["cue_separation_enforced"] is True
    assert validation["cue_separation_valid"] is True
    assert validation["validation_errors"] == []
    assert validation["template_bank_hash"] != balanced["template_bank_hash"]


def test_crosswired_stress_profile_is_explicitly_marked() -> None:
    set_heuristic_template_mode("multi", seed=0)
    set_heuristic_template_profile("crosswired_stress")

    validation = validate_active_heuristic_templates(
        expected_profile="crosswired_stress",
        expected_mode="multi",
        expected_seed=0,
    )

    assert validation["is_valid"] is True
    assert validation["config_matches"] is True
    assert validation["template_profile"] == "crosswired_stress"
    assert validation["template_profile_kind"] == "crosswired_stress"
    assert validation["cue_separation_enforced"] is False
    assert validation["cue_separation_valid"] is False
    assert validation["validation_errors"] == []
    assert validation["validation_warnings"] == [
        "Cue-separation checks are intentionally disabled for crosswired_stress."
    ]


def test_generate_latex_macros_emits_template_profile_and_gradient_range_macros() -> None:
    hds_test = {
        "accuracy": 0.987,
        "heuristic_match": 0.571,
        "by_target_heuristic": {
            "DD": {"detection_rate": 0.852},
            "OT": {"detection_rate": 0.604},
            "RC": {"detection_rate": 0.250},
        },
        "template_variability": {
            "DD": {"mean_within_problem_std": 0.11},
            "OT": {"mean_within_problem_std": 0.13},
            "RC": {"mean_within_problem_std": 0.15},
        },
    }
    hds_test_image = {
        "accuracy": 1.0,
        "heuristic_match": 0.558,
        "by_target_heuristic": {
            "DD": {"detection_rate": 0.815},
            "OT": {"detection_rate": 0.604},
            "RC": {"detection_rate": 0.250},
        },
        "template_variability": {
            "DD": {"mean_within_problem_std": 0.14},
            "OT": {"mean_within_problem_std": 0.15},
            "RC": {"mean_within_problem_std": 0.16},
        },
    }
    hds_test_style_mismatch = {
        "accuracy": 0.981,
        "heuristic_match": 0.565,
        "by_target_heuristic": {
            "DD": {"detection_rate": 0.815},
            "OT": {"detection_rate": 0.604},
            "RC": {"detection_rate": 0.269},
        },
        "template_variability": {
            "DD": {"mean_within_problem_std": 0.17},
            "OT": {"mean_within_problem_std": 0.18},
            "RC": {"mean_within_problem_std": 0.19},
        },
    }
    hds_test_image_style_mismatch = {
        "accuracy": 0.987,
        "heuristic_match": 0.500,
        "by_target_heuristic": {
            "DD": {"detection_rate": 0.648},
            "OT": {"detection_rate": 0.646},
            "RC": {"detection_rate": 0.212},
        },
        "template_variability": {
            "DD": {"mean_within_problem_std": 0.20},
            "OT": {"mean_within_problem_std": 0.21},
            "RC": {"mean_within_problem_std": 0.21},
        },
    }
    gradient_analysis = {
        "format": "effective_update_similarity",
        "similarities": {
            "DD-OT": 0.1219,
            "DD-RC": 0.1225,
            "OT-RC": 0.0965,
            "DD-STYLE": 0.1442,
            "OT-STYLE": 0.1131,
            "RC-STYLE": 0.1304,
        },
        "seed_control_summary": {
            "same_heuristic_seed_pairs": {
                "DD-DD_SEED123": 0.4112,
                "OT-OT_SEED123": 0.3888,
                "RC-RC_SEED123": 0.4015,
            },
            "primary_cross_pairs": {
                "DD-OT": 0.1219,
                "DD-RC": 0.1225,
                "OT-RC": 0.0965,
            },
            "same_heuristic_avg": 0.4005,
            "primary_cross_avg": 0.1136,
            "gap": 0.2869,
        },
    }

    macros = generate_latex_macros(
        hds_test=hds_test,
        hds_all=None,
        traps={},
        gradient_analysis=gradient_analysis,
        hds_test_image=hds_test_image,
        hds_test_style_mismatch=hds_test_style_mismatch,
        hds_test_image_style_mismatch=hds_test_image_style_mismatch,
        model_slug="Qwen3-VL-30B-A3B",
        include_global_macros=False,
        emit_contrastive_placeholders=False,
        emit_gradient_placeholders=True,
        emit_suffix_macros=False,
    )

    assert r"\newcommand{\TemplateBalancedHeuristicMatchThirtyB}{57.1\%}" in macros
    assert r"\newcommand{\TemplateBalancedMeanStdThirtyB}{0.1300}" in macros
    assert r"\newcommand{\TemplateStyleMismatchMeanStdImageThirtyB}{0.2067}" in macros
    assert r"\newcommand{\TemplateStyleMismatchHeuristicMatchThirtyB}{56.5\%}" in macros
    assert r"\newcommand{\TemplateStyleMismatchDDDetectionImageThirtyB}{64.8\%}" in macros
    assert r"\newcommand{\EffectiveUpdateMinCrossThirtyB}{0.10}" in macros
    assert r"\newcommand{\EffectiveUpdateMaxCrossThirtyB}{0.14}" in macros
    assert r"\newcommand{\EffectiveUpdateSameSeedAvgThirtyB}{0.4005}" in macros
    assert r"\newcommand{\EffectiveUpdatePrimaryCrossAvgThirtyB}{0.1136}" in macros
    assert r"\newcommand{\EffectiveUpdateSeedGapThirtyB}{0.2869}" in macros


def test_generate_template_variability_appendix_renders_expected_profiles(monkeypatch) -> None:
    fixtures = {
        ("text", None): {
            "template_variability": {
                "DD": {"mean_within_problem_std": 0.101},
                "OT": {"mean_within_problem_std": 0.202},
                "RC": {"mean_within_problem_std": 0.303},
            }
        },
        ("text", "style_mismatch"): {
            "template_variability": {
                "DD": {"mean_within_problem_std": 0.111},
                "OT": {"mean_within_problem_std": 0.222},
                "RC": {"mean_within_problem_std": 0.333},
            }
        },
        ("image", None): {
            "template_variability": {
                "DD": {"mean_within_problem_std": 0.121},
                "OT": {"mean_within_problem_std": 0.232},
                "RC": {"mean_within_problem_std": 0.343},
            }
        },
        ("image", "style_mismatch"): {
            "template_variability": {
                "DD": {"mean_within_problem_std": 0.131},
                "OT": {"mean_within_problem_std": 0.242},
                "RC": {"mean_within_problem_std": 0.353},
            }
        },
    }

    def fake_load_analysis(dataset, split=None, modality="text", model_slug=None, output_tag=None, **kwargs):
        assert dataset == "HDSv2"
        assert split == "test"
        return fixtures[(modality, output_tag)]

    monkeypatch.setattr("analysis.GenerateResultsFigures.load_analysis", fake_load_analysis)
    appendix = generate_template_variability_appendix(
        model_slug="Qwen3-VL-30B-A3B",
        probe_hds_dataset="HDSv2",
    )

    assert r"\label{tab:template-variability}" in appendix
    assert "Balanced Text" in appendix
    assert "Style-Mismatch Image" in appendix
    assert "0.2020" in appendix
    assert "0.2420" in appendix


def test_generate_embedding_results_appendix_renders_expected_macros() -> None:
    appendix = generate_embedding_results_appendix(model_slug="Qwen3-VL-30B-A3B")

    assert r"\label{tab:embedding-fingerprint}" in appendix
    assert r"\HDSTestTraceEmbedDDCountImageThirtyB{}" in appendix
    assert r"\HDSTestTraceEmbedSTYLECountImageThirtyB{}" in appendix
    assert r"\HDSTestResolvedTraceEmbedCoverageImageAltThirtyB{}" in appendix
    assert r"\HDSTestRCTraceEmbedDetectionImageAltThirtyB{}" in appendix


def test_appendix_generators_can_omit_canonical_labels(monkeypatch) -> None:
    def fake_load_analysis(dataset, split=None, modality="text", model_slug=None, output_tag=None, **kwargs):
        assert dataset == "HDSv2"
        assert split == "test"
        return {
            "template_variability": {
                "DD": {"mean_within_problem_std": 0.11},
                "OT": {"mean_within_problem_std": 0.22},
                "RC": {"mean_within_problem_std": 0.33},
            }
        }

    def fake_load_nudge_results_csv(split, modality="text", model_slug=None):
        assert split == "test"
        rows = []
        for index, target in enumerate(("RC", "DD", "OT"), start=1):
            rows.append(
                {
                    "hds_id": f"h{index}",
                    "a": str(index),
                    "b": str(index + 1),
                    "target_heuristic": target,
                    "lora": "RC",
                    "lora_correctness": "correct",
                    "lora_detected": "OT",
                    "base_correctness": "correct",
                    "base_detected": "DD" if modality == "text" else "RC",
                }
            )
        return rows

    monkeypatch.setattr("analysis.GenerateResultsFigures.load_analysis", fake_load_analysis)
    monkeypatch.setattr("analysis.GenerateResultsFigures.load_nudge_results_csv", fake_load_nudge_results_csv)
    monkeypatch.setattr(
        "analysis.GenerateResultsFigures.load_weight_similarity_analysis",
        lambda model_slug=None: {"similarities": {"DD-OT": 0.10, "DD-RC": 0.12, "OT-RC": 0.14}},
    )
    monkeypatch.setattr(
        "analysis.GenerateResultsFigures.load_gradient_analysis",
        lambda model_slug=None: {
            "seed_control_summary": {
                "same_heuristic_seed_pairs": {"DD-DD_SEED123": 0.44},
                "primary_cross_pairs": {"DD-OT": 0.10},
                "same_heuristic_avg": 0.44,
                "primary_cross_avg": 0.10,
                "gap": 0.34,
            }
        },
    )

    nudge_appendix = generate_nudge_examples(model_slug="Qwen3-VL-30B-A3B", include_labels=False)
    template_appendix = generate_template_variability_appendix(
        model_slug="Qwen3-VL-30B-A3B",
        include_labels=False,
        probe_hds_dataset="HDSv2",
    )
    embedding_appendix = generate_embedding_results_appendix(
        model_slug="Qwen3-VL-30B-A3B",
        include_labels=False,
    )
    similarity_appendix = generate_similarity_matrix(
        model_slug="Qwen3-VL-30B-A3B",
        include_labels=False,
    )

    assert r"\label{tab:nudge-examples}" not in nudge_appendix
    assert r"\label{tab:template-variability}" not in template_appendix
    assert r"\label{tab:embedding-fingerprint}" not in embedding_appendix
    assert r"\label{tab:similarity-matrix}" not in similarity_appendix
    assert r"\label{tab:similarity-seed-controls}" not in similarity_appendix


def test_generate_latex_macros_supports_hdsv2_family_match_breakdown() -> None:
    hds_test = {
        "accuracy": 0.521,
        "family_match_rate": 0.319,
        "by_design_family": {
            "DD": {"family_match_rate": 0.636},
            "OT": {"family_match_rate": 0.216},
            "RC": {"family_match_rate": 0.143},
        },
        "template_variability": {
            "DD": {"mean_within_problem_std": 0.11},
            "OT": {"mean_within_problem_std": 0.13},
            "RC": {"mean_within_problem_std": 0.15},
        },
    }
    hds_test_image = {
        "accuracy": 0.486,
        "family_match_rate": 0.042,
        "by_design_family": {
            "DD": {"family_match_rate": 0.114},
            "OT": {"family_match_rate": 0.020},
            "RC": {"family_match_rate": 0.0},
        },
        "template_variability": {
            "DD": {"mean_within_problem_std": 0.14},
            "OT": {"mean_within_problem_std": 0.15},
            "RC": {"mean_within_problem_std": 0.16},
        },
    }
    hds_test_style_mismatch = {
        "accuracy": 0.500,
        "family_match_rate": 0.312,
        "by_design_family": {
            "DD": {"family_match_rate": 0.955},
            "OT": {"family_match_rate": 0.059},
            "RC": {"family_match_rate": 0.0},
        },
        "template_variability": {
            "DD": {"mean_within_problem_std": 0.17},
            "OT": {"mean_within_problem_std": 0.18},
            "RC": {"mean_within_problem_std": 0.19},
        },
    }
    hds_test_image_style_mismatch = {
        "accuracy": 0.521,
        "family_match_rate": 0.042,
        "by_design_family": {
            "DD": {"family_match_rate": 0.114},
            "OT": {"family_match_rate": 0.020},
            "RC": {"family_match_rate": 0.0},
        },
        "template_variability": {
            "DD": {"mean_within_problem_std": 0.20},
            "OT": {"mean_within_problem_std": 0.21},
            "RC": {"mean_within_problem_std": 0.21},
        },
    }
    gradient_analysis = {
        "format": "effective_update_similarity",
        "similarities": {
            "DD-OT": 0.1219,
            "DD-RC": 0.1225,
            "OT-RC": 0.0965,
        },
    }

    macros = generate_latex_macros(
        hds_test=hds_test,
        hds_all=None,
        traps={},
        gradient_analysis=gradient_analysis,
        hds_test_image=hds_test_image,
        hds_test_style_mismatch=hds_test_style_mismatch,
        hds_test_image_style_mismatch=hds_test_image_style_mismatch,
        model_slug="Qwen3-VL-30B-A3B",
        include_global_macros=False,
        emit_contrastive_placeholders=False,
        emit_gradient_placeholders=True,
        emit_suffix_macros=False,
        probe_hds_dataset_name="HDSv2",
    )

    assert r"\newcommand{\TemplateBalancedDDDetectionThirtyB}{63.6\%}" in macros
    assert r"\newcommand{\TemplateStyleMismatchDDDetectionThirtyB}{95.5\%}" in macros
    assert r"\newcommand{\TemplateStyleMismatchDDDetectionImageThirtyB}{11.4\%}" in macros


def test_generate_latex_macros_emits_trap_target_support_macros() -> None:
    traps = {
        "accuracy": 0.5,
        "avg_perplexity": {"DD": 1.0, "OT": 2.0, "RC": 3.0},
        "by_design_family": {
            "DD": {"family_match_rate": 0.25},
            "OT": {"family_match_rate": 0.375},
            "RC": {"family_match_rate": 0.286},
        },
        "soft_target_stats": {
            "DD": {"target_support": 0.284, "target_support_se": 0.008},
            "OT": {"target_support": 0.302, "target_support_se": 0.014},
            "RC": {"target_support": 0.273, "target_support_se": 0.017},
        },
    }
    traps_image = {
        "accuracy": 0.5,
        "avg_perplexity": {"DD": 1.5, "OT": 2.5, "RC": 3.5},
        "by_design_family": {
            "DD": {"family_match_rate": 1.0},
            "OT": {"family_match_rate": 0.5},
            "RC": {"family_match_rate": 0.0},
        },
        "soft_target_stats": {
            "DD": {"target_support": 0.233, "target_support_se": 0.007},
            "OT": {"target_support": 0.179, "target_support_se": 0.017},
            "RC": {"target_support": 0.113, "target_support_se": 0.006},
        },
    }

    macros = generate_latex_macros(
        hds_test={},
        hds_all=None,
        traps=traps,
        traps_image=traps_image,
        model_slug="Qwen3-VL-30B-A3B",
        include_global_macros=False,
        emit_contrastive_placeholders=False,
        emit_gradient_placeholders=False,
        emit_suffix_macros=False,
    )

    assert r"\newcommand{\TrapsDDTargetSupportThirtyB}{28.4\%}" in macros
    assert r"\newcommand{\TrapsOTTargetSupportThirtyB}{30.2\%}" in macros
    assert r"\newcommand{\TrapsRCTargetSupportSEThirtyB}{1.7\%}" in macros
    assert r"\newcommand{\TrapsDDTargetSupportImageThirtyB}{23.3\%}" in macros
    assert r"\newcommand{\TrapsOTTargetSupportImageSEThirtyB}{1.7\%}" in macros
    assert r"\newcommand{\TrapsRCTargetSupportImageThirtyB}{11.3\%}" in macros


def test_generate_template_variability_appendix_rejects_invalid_image_quality(monkeypatch) -> None:
    fixtures = {
        ("text", None): {"template_variability": {"DD": {"mean_within_problem_std": 0.101}}},
        ("text", "style_mismatch"): {"template_variability": {"DD": {"mean_within_problem_std": 0.111}}},
        ("image", None): {
            "template_variability": {"DD": {"mean_within_problem_std": 0.121}},
            "quality_status": "invalid",
            "unknown_rate": 0.799,
            "unknown_count": 115,
            "total": 144,
        },
        ("image", "style_mismatch"): {
            "template_variability": {"DD": {"mean_within_problem_std": 0.131}},
            "quality_status": "invalid",
            "unknown_rate": 0.799,
            "unknown_count": 115,
            "total": 144,
        },
    }

    def fake_load_analysis(dataset, split=None, modality="text", model_slug=None, output_tag=None, **kwargs):
        return fixtures[(modality, output_tag)]

    monkeypatch.setattr("analysis.GenerateResultsFigures.load_analysis", fake_load_analysis)
    with pytest.raises(ValueError, match="too degraded"):
        generate_template_variability_appendix(
            model_slug="Qwen3-VL-30B-A3B",
            probe_hds_dataset="HDSv2",
        )


def test_load_analysis_strict_dataset_rejects_alias_fallback(tmp_path, monkeypatch) -> None:
    alias_dir = tmp_path / "fingerprint_hds_test_Qwen3-VL-30B-A3B"
    alias_dir.mkdir()
    (alias_dir / "fingerprint_analysis.json").write_text(
        '{"total": 1, "accuracy": 1.0, "heuristic_match": 1.0, "confusion_matrix": {}}'
    )
    (alias_dir / "fingerprint_results.csv").write_text(
        "hds_id,loss_best_heuristic,detected_heuristic\nh1,DD,DD\n"
    )

    monkeypatch.setattr("analysis.GenerateResultsFigures.RESULTS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="alias artifact exists"):
        load_analysis(
            "HDSv2",
            split="test",
            modality="text",
            model_slug="Qwen3-VL-30B-A3B",
            allow_alias_fallback=False,
        )


def test_analyze_results_records_quality_metrics() -> None:
    def make_result(hds_id: str, detected: str, losses: dict[str, float], neutral_loss: float) -> FingerprintingResult:
        return FingerprintingResult(
            hds_id=hds_id,
            a=12,
            b=34,
            product=408,
            target_heuristic="DD",
            design_family="DD",
            canonical_target_heuristic="DD",
            detected_heuristic=detected,
            detection_confidence=0.0,
            model_answer=408,
            is_correct=True,
            error_delta=0,
            error_heuristic=None,
            error_confidence=None,
            perplexity_losses=losses,
            trace=None,
            trace_heuristic=None,
            trace_confidence=None,
            neutral_loss=neutral_loss,
        )

    analysis = analyze_results(
        [
            make_result("h1", "UNKNOWN", {"DD": float("inf"), "OT": float("inf"), "RC": float("inf")}, float("inf")),
            make_result("h2", "UNKNOWN", {"DD": float("inf"), "OT": float("inf"), "RC": float("inf")}, float("inf")),
            make_result("h3", "DD", {"DD": 1.0, "OT": 2.0, "RC": 3.0}, 4.0),
        ]
    )

    assert analysis["unknown_count"] == 2
    assert analysis["unknown_rate"] == pytest.approx(2 / 3)
    assert analysis["nonfinite_loss_count"] == 2
    assert analysis["nonfinite_loss_rate"] == pytest.approx(2 / 3)
    assert analysis["quality_status"] == "invalid"


def test_generate_similarity_matrix_includes_seed_control_summary(monkeypatch) -> None:
    monkeypatch.setattr(
        "analysis.GenerateResultsFigures.load_weight_similarity_analysis",
        lambda model_slug=None: {
            "similarities": {
                "DD-OT": 0.10,
                "DD-RC": 0.12,
                "OT-RC": 0.14,
            }
        },
    )
    monkeypatch.setattr(
        "analysis.GenerateResultsFigures.load_gradient_analysis",
        lambda model_slug=None: {
            "seed_control_summary": {
                "same_heuristic_seed_pairs": {
                    "DD-DD_SEED123": 0.44,
                    "OT-OT_SEED123": 0.41,
                    "RC-RC_SEED123": 0.39,
                },
                "primary_cross_pairs": {
                    "DD-OT": 0.10,
                    "DD-RC": 0.12,
                    "OT-RC": 0.14,
                },
                "same_heuristic_avg": 0.4133,
                "primary_cross_avg": 0.12,
                "gap": 0.2933,
            }
        },
    )

    appendix = generate_similarity_matrix(model_slug="Qwen3-VL-30B-A3B")

    assert r"\label{tab:similarity-seed-controls}" in appendix
    assert "Same-heuristic average & 0.4133" in appendix
    assert "Primary cross-heuristic average & 0.1200" in appendix
    assert "Gap & 0.2933" in appendix


def test_merge_nudge_appendix_fragments_creates_single_canonical_label(tmp_path) -> None:
    fragment_30b = tmp_path / "appendix_nudge_examples.30b.tex"
    fragment_235b = tmp_path / "appendix_nudge_examples.235b.tex"
    merged = tmp_path / "appendix_nudge_examples.tex"

    fragment_30b.write_text(
        "% Auto-generated nudge test examples (text vs image)\n"
        "% Model: 30B\n\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{LoRA Nudge Test: Text vs Image Modality (lowest-loss heuristic)}\n"
        "\\label{tab:nudge-examples}\n"
        "\\small\n"
        "\\begin{tabular}{@{}lccccc@{}}\n"
        "\\toprule\n"
        "Problem & Target & \\multicolumn{2}{c}{Text} & \\multicolumn{2}{c}{Image} \\\\\n"
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}\n"
        "        &        & Base & LoRA & Base & LoRA \\\\\n"
        "\\midrule\n"
        "$1 \\times 2$ & RC & \\checkmark (RC) & \\checkmark (OT) & \\checkmark (RC) & \\checkmark (OT) \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    fragment_235b.write_text(
        "% Auto-generated nudge test examples (text vs image)\n"
        "% Model: 235B\n\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{LoRA Nudge Test: Text vs Image Modality (lowest-loss heuristic)}\n"
        "\\label{tab:nudge-examples}\n"
        "\\small\n"
        "\\begin{tabular}{@{}lccccc@{}}\n"
        "\\toprule\n"
        "Problem & Target & \\multicolumn{2}{c}{Text} & \\multicolumn{2}{c}{Image} \\\\\n"
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}\n"
        "        &        & Base & LoRA & Base & LoRA \\\\\n"
        "\\midrule\n"
        "$3 \\times 4$ & DD & \\checkmark (DD) & \\texttimes (OT) & \\checkmark (DD) & \\checkmark (OT) \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    merge_nudge_appendix_fragments([fragment_30b, fragment_235b], merged)
    merged_text = merged.read_text()

    assert merged_text.count(r"\label{tab:nudge-examples}") == 1
    assert r"\textbf{30B}" in merged_text
    assert r"\textbf{235B}" in merged_text
    assert "$1 \\times 2$ & RC" in merged_text
    assert "$3 \\times 4$ & DD" in merged_text


def test_merge_similarity_appendix_fragments_creates_single_canonical_labels(tmp_path) -> None:
    fragment_30b = tmp_path / "appendix_similarity_matrix.30b.tex"
    fragment_235b = tmp_path / "appendix_similarity_matrix.235b.tex"
    merged = tmp_path / "appendix_similarity_matrix.tex"

    fragment_30b.write_text(
        "% Auto-generated cosine similarity matrix\n"
        "% Model: 30B\n\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Adapter Weight Cosine Similarity Matrix}\n"
        "\\label{tab:similarity-matrix}\n"
        "\\begin{tabular}{@{}lccc@{}}\n"
        "\\toprule\n"
        " & DD & OT & RC \\\\\n"
        "\\midrule\n"
        "DD & 1.0000 & 0.1000 & 0.2000 \\\\\n"
        "OT & 0.1000 & 1.0000 & 0.3000 \\\\\n"
        "RC & 0.2000 & 0.3000 & 1.0000 \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n\n"
        "% Average off-diagonal similarity: 0.2000\n"
        "% Values near 0 indicate orthogonal (independent) adapter directions\n\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Effective-Update Seed-Control Summary}\n"
        "\\label{tab:similarity-seed-controls}\n"
        "\\begin{tabular}{@{}lc@{}}\n"
        "\\toprule\n"
        "Pair & Cosine \\\\\n"
        "\\midrule\n"
        "DD-DD seed 123 & 0.4100 \\\\\n"
        "\\midrule\n"
        "Same-heuristic average & 0.4100 \\\\\n"
        "Primary cross-heuristic average & 0.2000 \\\\\n"
        "Gap & 0.2100 \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n\n"
        "% Primary cross-heuristic baseline pairs: DD-OT=0.1000\n"
    )
    fragment_235b.write_text(
        "% Auto-generated cosine similarity matrix\n"
        "% Model: 235B\n\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Adapter Weight Cosine Similarity Matrix}\n"
        "\\label{tab:similarity-matrix}\n"
        "\\begin{tabular}{@{}lccc@{}}\n"
        "\\toprule\n"
        " & DD & OT & RC \\\\\n"
        "\\midrule\n"
        "DD & 1.0000 & 0.0100 & 0.0200 \\\\\n"
        "OT & 0.0100 & 1.0000 & 0.0300 \\\\\n"
        "RC & 0.0200 & 0.0300 & 1.0000 \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n\n"
        "% Average off-diagonal similarity: 0.0200\n"
        "% Values near 0 indicate orthogonal (independent) adapter directions\n\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Effective-Update Seed-Control Summary}\n"
        "\\label{tab:similarity-seed-controls}\n"
        "\\begin{tabular}{@{}lc@{}}\n"
        "\\toprule\n"
        "Pair & Cosine \\\\\n"
        "\\midrule\n"
        "DD-DD seed 123 & 0.0800 \\\\\n"
        "\\midrule\n"
        "Same-heuristic average & 0.0800 \\\\\n"
        "Primary cross-heuristic average & 0.0200 \\\\\n"
        "Gap & 0.0600 \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n\n"
        "% Primary cross-heuristic baseline pairs: DD-OT=0.0100\n"
    )

    merge_similarity_appendix_fragments([fragment_30b, fragment_235b], merged)
    merged_text = merged.read_text()

    assert merged_text.count(r"\label{tab:similarity-matrix}") == 1
    assert merged_text.count(r"\label{tab:similarity-seed-controls}") == 1
    assert r"\textbf{30B}" in merged_text
    assert r"\textbf{235B}" in merged_text
    assert "DD & 1.0000 & 0.1000 & 0.2000" in merged_text
    assert "DD & 1.0000 & 0.0100 & 0.0200" in merged_text
    assert "% 30B average off-diagonal similarity: 0.2000" in merged_text
    assert "% 235B primary cross-heuristic baseline pairs: DD-OT=0.0100" in merged_text


def test_merge_template_variability_appendix_fragments_creates_single_canonical_label(tmp_path) -> None:
    fragment_30b = tmp_path / "appendix_template_variability.30b.tex"
    fragment_235b = tmp_path / "appendix_template_variability.235b.tex"
    merged = tmp_path / "appendix_template_variability.tex"

    fragment_30b.write_text(
        "% Auto-generated template-variability appendix\n"
        "% Model: 30B\n\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Template-variability robustness summary}\n"
        "\\label{tab:template-variability}\n"
        "\\begin{tabular}{@{}lcccc@{}}\n"
        "\\toprule\n"
        "Profile & DD std & OT std & RC std & Mean std \\\\\n"
        "\\midrule\n"
        "Balanced Text & 0.1000 & 0.2000 & 0.3000 & 0.2000 \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    fragment_235b.write_text(
        "% Auto-generated template-variability appendix\n"
        "% Model: 235B\n\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Template-variability robustness summary}\n"
        "\\label{tab:template-variability}\n"
        "\\begin{tabular}{@{}lcccc@{}}\n"
        "\\toprule\n"
        "Profile & DD std & OT std & RC std & Mean std \\\\\n"
        "\\midrule\n"
        "Balanced Text & 0.4000 & 0.5000 & 0.6000 & 0.5000 \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    merge_template_variability_appendix_fragments([fragment_30b, fragment_235b], merged)
    merged_text = merged.read_text()

    assert merged_text.count(r"\label{tab:template-variability}") == 1
    assert r"\textbf{30B}" in merged_text
    assert r"\textbf{235B}" in merged_text
    assert "Balanced Text & 0.1000 & 0.2000 & 0.3000 & 0.2000" in merged_text
    assert "Balanced Text & 0.4000 & 0.5000 & 0.6000 & 0.5000" in merged_text


def test_merge_embedding_results_appendix_fragments_creates_single_canonical_label(tmp_path) -> None:
    fragment_30b = tmp_path / "appendix_embedding_results.30b.tex"
    fragment_235b = tmp_path / "appendix_embedding_results.235b.tex"
    merged = tmp_path / "appendix_embedding_results.tex"

    fragment_30b.write_text(generate_embedding_results_appendix(model_slug="Qwen3-VL-30B-A3B"))
    fragment_235b.write_text(generate_embedding_results_appendix(model_slug="Qwen3-VL-235B-A22B"))

    merge_embedding_results_appendix_fragments([fragment_30b, fragment_235b], merged)
    merged_text = merged.read_text()

    assert merged_text.count(r"\label{tab:embedding-fingerprint}") == 1
    assert r"\textbf{30B}" in merged_text
    assert r"\textbf{235B}" in merged_text
    assert r"\HDSTestTraceEmbedSTYLECountImageThirtyB{}" in merged_text
    assert r"\HDSTestTraceEmbedSTYLECountImageTwoThirtyFiveB{}" in merged_text
    assert r"\HDSTestRCTraceEmbedDetectionImageAltTwoThirtyFiveB{}" in merged_text
    assert "DD-targeted items" in merged_text
    assert "All image traces resolved" in merged_text


def test_merge_text_fragments_dedupes_repeated_newcommands(tmp_path) -> None:
    first = tmp_path / "first.tex"
    second = tmp_path / "second.tex"
    merged = tmp_path / "merged.tex"

    first.write_text(
        "% fragment 1\n"
        "\\newcommand{\\DatasetSplitRatios}{70/15/15}\n"
        "\\newcommand{\\OnlyFirst}{A}\n"
    )
    second.write_text(
        "% fragment 2\n"
        "\\newcommand{\\DatasetSplitRatios}{70/15/15}\n"
        "\\newcommand{\\OnlySecond}{B}\n"
    )

    merge_text_fragments([first, second], merged, dedupe_newcommands=True)
    merged_text = merged.read_text()

    assert merged_text.count(r"\newcommand{\DatasetSplitRatios}{70/15/15}") == 1
    assert r"\newcommand{\OnlyFirst}{A}" in merged_text
    assert r"\newcommand{\OnlySecond}{B}" in merged_text
