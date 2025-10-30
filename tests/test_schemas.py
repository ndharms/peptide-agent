"""Tests for schema validation and parsing."""

import pytest

from peptide_agent.schemas import (
    BatchPredictionResult,
    IntervalBound,
    PeptideSynthesisConditions,
)


class TestIntervalBound:
    """Test IntervalBound parsing and serialization."""

    def test_from_string_closed(self):
        """Test parsing closed interval."""
        interval = IntervalBound.from_string("[5.0,7.0]")
        assert interval.lower == 5.0
        assert interval.upper == 7.0
        assert interval.lower_inclusive is True
        assert interval.upper_inclusive is True

    def test_from_string_open(self):
        """Test parsing open interval."""
        interval = IntervalBound.from_string("(5.0,7.0)")
        assert interval.lower == 5.0
        assert interval.upper == 7.0
        assert interval.lower_inclusive is False
        assert interval.upper_inclusive is False

    def test_from_string_mixed(self):
        """Test parsing mixed interval."""
        interval = IntervalBound.from_string("(5.0,7.0]")
        assert interval.lower == 5.0
        assert interval.upper == 7.0
        assert interval.lower_inclusive is False
        assert interval.upper_inclusive is True

    def test_from_string_negative(self):
        """Test parsing with negative numbers."""
        interval = IntervalBound.from_string("(-3.0,-1.0)")
        assert interval.lower == -3.0
        assert interval.upper == -1.0

    def test_from_string_invalid(self):
        """Test parsing invalid interval."""
        with pytest.raises(ValueError):
            IntervalBound.from_string("invalid")

    def test_to_string(self):
        """Test serialization to string."""
        interval = IntervalBound(lower=5.0, upper=7.0, lower_inclusive=True, upper_inclusive=False)
        assert interval.to_string() == "[5.0,7.0)"


class TestPeptideSynthesisConditions:
    """Test PeptideSynthesisConditions validation and parsing."""

    def test_from_report_string_valid(self):
        """Test parsing valid report string."""
        report = """PH: (5.0,7.0)
Concentration (log M): (-3.0,-1.0)
Temperature (C): (20.0,25.0)
Solvent: Water
Estimated Time (minutes): (60,120)"""

        conditions = PeptideSynthesisConditions.from_report_string(report)
        assert conditions.ph.lower == 5.0
        assert conditions.ph.upper == 7.0
        assert conditions.concentration_log_m.lower == -3.0
        assert conditions.solvent == "Water"
        assert conditions.time_minutes.lower == 60

    def test_from_report_string_with_reasoning(self):
        """Test parsing report with reasoning."""
        report = """PH: (5.0,7.0)
Concentration (log M): (-3.0,-1.0)
Temperature (C): (20.0,25.0)
Solvent: Water
Estimated Time (minutes): (60,120)

Reasoning: This is based on similar peptides."""

        conditions = PeptideSynthesisConditions.from_report_string(report)
        assert "similar peptides" in conditions.reasoning

    def test_to_report_string(self):
        """Test serialization to report string."""
        conditions = PeptideSynthesisConditions(
            ph=IntervalBound(lower=5.0, upper=7.0, lower_inclusive=True, upper_inclusive=True),
            concentration_log_m=IntervalBound(
                lower=-3.0, upper=-1.0, lower_inclusive=True, upper_inclusive=True
            ),
            temperature_c=IntervalBound(
                lower=20.0, upper=25.0, lower_inclusive=True, upper_inclusive=True
            ),
            solvent="Water",
            time_minutes=IntervalBound(
                lower=60, upper=120, lower_inclusive=True, upper_inclusive=True
            ),
            reasoning="Test reasoning",
        )

        report = conditions.to_report_string()
        assert "PH: [5.0,7.0]" in report
        assert "Solvent: Water" in report
        assert "Reasoning: Test reasoning" in report

    def test_validate_solvent_empty(self):
        """Test that empty solvent raises error."""
        with pytest.raises(ValueError):
            PeptideSynthesisConditions(
                ph=IntervalBound(lower=5.0, upper=7.0),
                concentration_log_m=IntervalBound(lower=-3.0, upper=-1.0),
                temperature_c=IntervalBound(lower=20.0, upper=25.0),
                solvent="",
                time_minutes=IntervalBound(lower=60, upper=120),
            )


class TestBatchPredictionResult:
    """Test BatchPredictionResult model."""

    def test_batch_result_success(self):
        """Test successful batch result."""
        result = BatchPredictionResult(
            id=0,
            peptide_code="FF",
            target_structural_assembly="nanofibers",
            conditions=PeptideSynthesisConditions(
                ph=IntervalBound(lower=5.0, upper=7.0),
                concentration_log_m=IntervalBound(lower=-3.0, upper=-1.0),
                temperature_c=IntervalBound(lower=20.0, upper=25.0),
                solvent="Water",
                time_minutes=IntervalBound(lower=60, upper=120),
            ),
            raw_report="PH: (5.0,7.0)...",
        )

        assert result.id == 0
        assert result.peptide_code == "FF"
        assert result.error is None

    def test_batch_result_error(self):
        """Test batch result with error."""
        result = BatchPredictionResult(
            id=1,
            peptide_code="XX",
            target_structural_assembly="unknown",
            conditions=None,
            error="Failed to generate prediction",
        )

        assert result.id == 1
        assert result.conditions is None
        assert result.error is not None
