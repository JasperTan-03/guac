"""Tests for guac.evaluation.evaluator.

Covers pure parsing/formatting functions, mocked batch inference,
mocked benchmark evaluation loops, and the end-to-end run_evaluation
orchestrator.  All tests run on CPU (no GPU required).
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from guac.evaluation.evaluator import (
    _extract_pil_image,
    evaluate_mathvista,
    evaluate_mmmu,
    format_mathvista_prompt,
    format_mmmu_prompt,
    parse_mc_answer,
    parse_numeric_answer,
    run_evaluation,
    run_inference_batch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_image(mode: str = "RGB") -> Image.Image:
    return Image.new(mode, (4, 4), color="red")


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    _tiny_image().save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# parse_mc_answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", None),
        (None, None),
        ("The answer is B", "B"),
        ("the answer is b", "B"),
        ("Option: C", "C"),
        ("A.", "A"),
        ("**D**", "D"),
        ("I think B is correct", "B"),
        ("<think>Let me pick A</think>The answer is C", "C"),
        ("I have no idea", None),
    ],
    ids=[
        "empty",
        "none",
        "prefix-B",
        "prefix-lowercase",
        "option-C",
        "dotted-A",
        "bold-D",
        "standalone-B",
        "think-tag-stripped",
        "no-letter",
    ],
)
def test_parse_mc_answer(text, expected):
    result = parse_mc_answer(text) if text is not None else parse_mc_answer("")
    if text is None:
        assert result is None
    else:
        assert result == expected


# ---------------------------------------------------------------------------
# parse_numeric_answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", None),
        ("\\boxed{42}", "42"),
        ("\\boxed{1} then \\boxed{2}", "2"),
        ("The answer is 3.14", "3.14"),
        ("-7", "-7"),
        ("<think>5</think>The answer is 3", "3"),
        ("no numbers here", None),
    ],
    ids=[
        "empty",
        "boxed-single",
        "boxed-last-wins",
        "plain-decimal",
        "negative",
        "think-tag-stripped",
        "no-numbers",
    ],
)
def test_parse_numeric_answer(text, expected):
    assert parse_numeric_answer(text) == expected


# ---------------------------------------------------------------------------
# format_mathvista_prompt
# ---------------------------------------------------------------------------


class TestFormatMathvistaPrompt:
    def test_with_choices(self):
        result = format_mathvista_prompt("What is X?", ["A", "B", "C"])
        assert "Choices:" in result
        assert "option letter" in result.lower()

    def test_without_choices(self):
        result = format_mathvista_prompt("What is X?", None)
        assert "numerical" in result.lower()

    def test_empty_choices_treated_as_none(self):
        result = format_mathvista_prompt("What is X?", [])
        assert "numerical" in result.lower()


# ---------------------------------------------------------------------------
# format_mmmu_prompt
# ---------------------------------------------------------------------------


class TestFormatMmmuPrompt:
    def test_with_options(self):
        result = format_mmmu_prompt("What is X?", ["opt1", "opt2"])
        assert "Options:" in result
        assert "single letter" in result.lower()

    def test_includes_question(self):
        result = format_mmmu_prompt("My question?", ["A"])
        assert "My question?" in result


# ---------------------------------------------------------------------------
# _extract_pil_image
# ---------------------------------------------------------------------------


class TestExtractPilImage:
    def test_none(self):
        assert _extract_pil_image(None) is None

    def test_pil_image_converted_to_rgb(self):
        rgba = _tiny_image("RGBA")
        result = _extract_pil_image(rgba)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_dict_with_bytes(self):
        result = _extract_pil_image({"bytes": _png_bytes()})
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_dict_with_path(self, tmp_path):
        p = tmp_path / "img.png"
        _tiny_image().save(p)
        result = _extract_pil_image({"path": str(p)})
        assert isinstance(result, Image.Image)

    def test_dict_invalid_bytes(self):
        assert _extract_pil_image({"bytes": b"not-an-image"}) is None

    def test_empty_dict(self):
        assert _extract_pil_image({}) is None

    def test_other_type(self):
        assert _extract_pil_image(12345) is None


# ---------------------------------------------------------------------------
# run_inference_batch  (mocked model + processor)
# ---------------------------------------------------------------------------


def _make_mock_model_and_processor(decoded_texts: list[str]):
    """Build mocks that simulate model.generate() and processor.batch_decode()."""
    batch_size = len(decoded_texts)

    param = torch.nn.Parameter(torch.zeros(1))  # CPU tensor
    model = MagicMock()
    model.parameters.return_value = iter([param])

    prompt_len = 5
    gen_len = 3
    total_len = prompt_len + gen_len
    model.generate.return_value = torch.zeros(batch_size, total_len, dtype=torch.long)

    processor = MagicMock()
    processor.apply_chat_template.return_value = "dummy prompt text"
    processor.return_value = {
        "input_ids": torch.zeros(batch_size, prompt_len, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, prompt_len, dtype=torch.long),
    }
    processor.batch_decode.return_value = decoded_texts

    return model, processor


class TestRunInferenceBatch:
    def test_single_item_no_images(self):
        model, processor = _make_mock_model_and_processor(["The answer is A"])
        items = [
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]}
                ]
            }
        ]
        results = run_inference_batch(model, processor, items)
        assert results == ["The answer is A"]
        model.generate.assert_called_once()

    def test_batch_with_images(self):
        model, processor = _make_mock_model_and_processor(["A", "B"])
        img = _tiny_image()
        items = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": "q1"},
                        ],
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": "q2"},
                        ],
                    }
                ]
            },
        ]
        results = run_inference_batch(model, processor, items)
        assert len(results) == 2
        # Processor should have been called with images
        call_kwargs = processor.call_args
        assert call_kwargs.kwargs.get("images") is not None

    def test_processor_error_returns_empty(self):
        model = MagicMock()
        model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
        processor = MagicMock()
        processor.apply_chat_template.return_value = "text"
        processor.side_effect = RuntimeError("boom")

        items = [
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "q"}]}
                ]
            }
        ]
        results = run_inference_batch(model, processor, items)
        assert results == [""]


# ---------------------------------------------------------------------------
# evaluate_mathvista  (mocked dataset + inference)
# ---------------------------------------------------------------------------


def _mathvista_cfg(batch_size: int = 2) -> OmegaConf:
    return OmegaConf.create(
        {
            "evaluation": {
                "batch_size": batch_size,
                "benchmarks": {
                    "mathvista": {"hf_id": "fake/mathvista", "split": "test"}
                },
            }
        }
    )


class TestEvaluateMathvista:
    @patch("guac.evaluation.evaluator.run_inference_batch")
    @patch("guac.evaluation.evaluator.load_dataset")
    def test_all_correct_mc(self, mock_load, mock_infer):
        img = _tiny_image()
        mock_load.return_value = [
            {"decoded_image": img, "question": "Q1", "choices": ["A", "B"], "answer": "A"},
            {"decoded_image": img, "question": "Q2", "choices": ["C", "D"], "answer": "B"},
        ]
        mock_infer.return_value = ["The answer is A", "The answer is B"]

        result = evaluate_mathvista(MagicMock(), MagicMock(), _mathvista_cfg())
        assert result["accuracy"] == 1.0
        assert result["correct"] == 2
        assert result["total"] == 2
        assert result["skipped"] == 0

    @patch("guac.evaluation.evaluator.run_inference_batch")
    @patch("guac.evaluation.evaluator.load_dataset")
    def test_mixed_mc_and_numeric(self, mock_load, mock_infer):
        img = _tiny_image()
        mock_load.return_value = [
            {"decoded_image": img, "question": "Q1", "choices": ["X", "Y"], "answer": "A"},
            {"decoded_image": img, "question": "Q2", "choices": None, "answer": "42"},
        ]
        # First correct (MC), second wrong (numeric)
        mock_infer.return_value = ["A", "99"]

        result = evaluate_mathvista(MagicMock(), MagicMock(), _mathvista_cfg())
        assert result["correct"] == 1
        assert result["accuracy"] == pytest.approx(0.5)

    @patch("guac.evaluation.evaluator.run_inference_batch")
    @patch("guac.evaluation.evaluator.load_dataset")
    def test_missing_image_skipped(self, mock_load, mock_infer):
        mock_load.return_value = [
            {"decoded_image": None, "image": None, "question": "Q", "choices": ["A"], "answer": "A"},
        ]
        mock_infer.return_value = []  # never called

        result = evaluate_mathvista(MagicMock(), MagicMock(), _mathvista_cfg())
        assert result["skipped"] == 1
        assert result["correct"] == 0

    @patch("guac.evaluation.evaluator.load_dataset", side_effect=Exception("network error"))
    def test_dataset_load_failure(self, _mock_load):
        result = evaluate_mathvista(MagicMock(), MagicMock(), _mathvista_cfg())
        assert result["accuracy"] == 0.0
        assert result["total"] == 0


# ---------------------------------------------------------------------------
# evaluate_mmmu  (mocked dataset + inference)
# ---------------------------------------------------------------------------


def _mmmu_cfg(batch_size: int = 2) -> OmegaConf:
    return OmegaConf.create(
        {
            "evaluation": {
                "batch_size": batch_size,
                "benchmarks": {
                    "mmmu": {"hf_id": "fake/mmmu", "config": "Math", "split": "val"}
                },
            }
        }
    )


class TestEvaluateMmmu:
    @patch("guac.evaluation.evaluator.run_inference_batch")
    @patch("guac.evaluation.evaluator.load_dataset")
    def test_all_correct(self, mock_load, mock_infer):
        img = _tiny_image()
        mock_load.return_value = [
            {"image_1": img, "question": "Q1", "options": ["a", "b", "c", "d"], "answer": "A"},
            {"image_1": img, "question": "Q2", "options": ["a", "b", "c", "d"], "answer": "C"},
        ]
        mock_infer.return_value = ["A", "C"]

        result = evaluate_mmmu(MagicMock(), MagicMock(), _mmmu_cfg())
        assert result["accuracy"] == 1.0
        assert result["correct"] == 2

    @patch("guac.evaluation.evaluator.run_inference_batch")
    @patch("guac.evaluation.evaluator.load_dataset")
    def test_string_encoded_options(self, mock_load, mock_infer):
        img = _tiny_image()
        mock_load.return_value = [
            {"image_1": img, "question": "Q", "options": "['opt1', 'opt2', 'opt3', 'opt4']", "answer": "B"},
        ]
        mock_infer.return_value = ["B"]

        result = evaluate_mmmu(MagicMock(), MagicMock(), _mmmu_cfg())
        assert result["correct"] == 1

    @patch("guac.evaluation.evaluator.run_inference_batch")
    @patch("guac.evaluation.evaluator.load_dataset")
    def test_no_image_proceeds_text_only(self, mock_load, mock_infer):
        mock_load.return_value = [
            {"image_1": None, "question": "Q", "options": ["a", "b"], "answer": "A"},
        ]
        mock_infer.return_value = ["A"]

        result = evaluate_mmmu(MagicMock(), MagicMock(), _mmmu_cfg())
        assert result["skipped"] == 0
        assert result["correct"] == 1

    @patch("guac.evaluation.evaluator.load_dataset")
    def test_dataset_load_fallback(self, mock_load):
        """First load with config raises, second without config succeeds."""
        img = _tiny_image()
        mock_load.side_effect = [
            Exception("config not found"),
            [{"image_1": img, "question": "Q", "options": ["a"], "answer": "A"}],
        ]

        with patch("guac.evaluation.evaluator.run_inference_batch", return_value=["A"]):
            result = evaluate_mmmu(MagicMock(), MagicMock(), _mmmu_cfg())

        assert mock_load.call_count == 2
        assert result["correct"] == 1


# ---------------------------------------------------------------------------
# run_evaluation  (end-to-end with all mocks)
# ---------------------------------------------------------------------------


def _full_cfg(tmp_path) -> OmegaConf:
    return OmegaConf.create(
        {
            "model": {"name": "fake/model"},
            "evaluation": {
                "output_path": str(tmp_path / "results.json"),
                "batch_size": 2,
                "benchmarks": {
                    "mathvista": {"hf_id": "fake", "split": "test"},
                    "mmmu": {"hf_id": "fake", "config": "Math", "split": "val"},
                },
                "mlflow_tracking_uri": str(tmp_path / "mlruns"),
                "mlflow_experiment": "test-eval",
            },
        }
    )


class TestRunEvaluation:
    @patch("guac.evaluation.evaluator.mlflow")
    @patch("guac.evaluation.evaluator.evaluate_mmmu")
    @patch("guac.evaluation.evaluator.evaluate_mathvista")
    @patch("guac.evaluation.evaluator.load_model_and_processor")
    def test_happy_path(self, mock_load_model, mock_mv, mock_mmmu, mock_mlflow, tmp_path):
        mock_load_model.return_value = (MagicMock(), MagicMock())
        mock_mv.return_value = {"accuracy": 0.8, "correct": 80, "total": 100, "skipped": 0}
        mock_mmmu.return_value = {"accuracy": 0.6, "correct": 60, "total": 100, "skipped": 0}

        cfg = _full_cfg(tmp_path)
        results = run_evaluation(cfg, "checkpoints/test")

        # Model loaded correctly
        mock_load_model.assert_called_once_with("checkpoints/test", "fake/model")

        # Both benchmarks called
        mock_mv.assert_called_once()
        mock_mmmu.assert_called_once()

        # Macro average
        assert results["macro_average"] == pytest.approx(0.7)

        # JSON output written
        out_path = tmp_path / "results.json"
        assert out_path.exists()
        saved = json.loads(out_path.read_text())
        assert saved["macro_average"] == pytest.approx(0.7)
        assert "MathVista" in saved["benchmarks"]
        assert "MMMU" in saved["benchmarks"]

    @patch("guac.evaluation.evaluator.mlflow")
    @patch("guac.evaluation.evaluator.evaluate_mmmu")
    @patch("guac.evaluation.evaluator.evaluate_mathvista", side_effect=RuntimeError("crash"))
    @patch("guac.evaluation.evaluator.load_model_and_processor")
    def test_one_benchmark_crash(self, mock_load_model, _mock_mv, mock_mmmu, mock_mlflow, tmp_path):
        mock_load_model.return_value = (MagicMock(), MagicMock())
        mock_mmmu.return_value = {"accuracy": 0.5, "correct": 50, "total": 100, "skipped": 0}

        cfg = _full_cfg(tmp_path)
        results = run_evaluation(cfg, "checkpoints/test")

        # MathVista should get fallback zeros
        assert results["benchmarks"]["MathVista"]["accuracy"] == 0.0
        # MMMU still runs
        assert results["benchmarks"]["MMMU"]["accuracy"] == 0.5
        # Macro average accounts for the zero
        assert results["macro_average"] == pytest.approx(0.25)
