"""Schema definitions for peptide agent outputs with validation."""

import re

from pydantic import BaseModel, Field, field_validator


class IntervalBound(BaseModel):
    """Represents an interval with numeric bounds and open/closed indicators."""

    lower: float = Field(..., description="Lower bound of the interval")
    upper: float = Field(..., description="Upper bound of the interval")
    lower_inclusive: bool = Field(True, description="Whether lower bound is inclusive")
    upper_inclusive: bool = Field(True, description="Whether upper bound is inclusive")

    def to_string(self) -> str:
        """Convert to interval notation string."""
        left = "[" if self.lower_inclusive else "("
        right = "]" if self.upper_inclusive else ")"
        return f"{left}{self.lower},{self.upper}{right}"

    @classmethod
    def from_string(cls, interval_str: str) -> "IntervalBound":
        """Parse interval string like (5,7) or [5,7] or (5,7] etc."""
        # More flexible pattern that allows decimals like .5 and various number formats
        pattern = r"^([\[\(])\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*([\]\)])$"
        match = re.match(pattern, interval_str.strip())
        if not match:
            raise ValueError(f"Invalid interval format: {interval_str}")

        left_bracket, lower, upper, right_bracket = match.groups()
        
        try:
            lower_val = float(lower)
            upper_val = float(upper)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Could not convert bounds to float in interval: {interval_str}") from e
        
        return cls(
            lower=lower_val,
            upper=upper_val,
            lower_inclusive=(left_bracket == "["),
            upper_inclusive=(right_bracket == "]"),
        )


class PeptideSynthesisConditions(BaseModel):
    """Validated experimental conditions for peptide synthesis."""

    ph: IntervalBound = Field(..., description="pH range")
    concentration_log_m: IntervalBound = Field(..., description="Concentration in log M")
    temperature_c: IntervalBound = Field(..., description="Temperature in Celsius")
    solvent: str = Field(..., description="Solvent choice")
    time_minutes: IntervalBound = Field(..., description="Estimated time in minutes")
    reasoning: str = Field(default="", description="Reasoning for conditions")

    @field_validator("solvent")
    @classmethod
    def validate_solvent(cls, v: str) -> str:
        """Ensure solvent is not empty."""
        if not v or not v.strip():
            raise ValueError("Solvent must be a non-empty string")
        return v.strip()

    def to_report_string(self) -> str:
        """Convert to standardized report format."""
        lines = [
            f"PH: {self.ph.to_string()}",
            f"Concentration (log M): {self.concentration_log_m.to_string()}",
            f"Temperature (C): {self.temperature_c.to_string()}",
            f"Solvent: {self.solvent}",
            f"Estimated Time (minutes): {self.time_minutes.to_string()}",
        ]
        if self.reasoning:
            lines.append(f"\nReasoning: {self.reasoning}")
        return "\n".join(lines)

    @classmethod
    def from_report_string(cls, report: str) -> "PeptideSynthesisConditions":
        """Parse a report string into structured conditions."""
        lines = [line.strip() for line in report.strip().split("\n") if line.strip()]

        data = {}
        reasoning_lines = []
        in_reasoning = False

        for line in lines:
            if line.startswith("Reasoning:"):
                in_reasoning = True
                reasoning_lines.append(line.replace("Reasoning:", "").strip())
            elif in_reasoning:
                reasoning_lines.append(line)
            elif ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                if key == "ph":
                    data["ph"] = IntervalBound.from_string(value)
                elif "concentration" in key:
                    data["concentration_log_m"] = IntervalBound.from_string(value)
                elif "temperature" in key:
                    data["temperature_c"] = IntervalBound.from_string(value)
                elif key == "solvent":
                    data["solvent"] = value
                elif "time" in key:
                    data["time_minutes"] = IntervalBound.from_string(value)

        if reasoning_lines:
            data["reasoning"] = " ".join(reasoning_lines)

        return cls(**data)


class BatchPredictionResult(BaseModel):
    """Result from batch prediction."""

    id: int = Field(..., description="Request ID")
    peptide_code: str = Field(..., description="Peptide code")
    target_structural_assembly: str = Field(..., description="Target morphology")
    conditions: PeptideSynthesisConditions | None = Field(None, description="Predicted conditions")
    raw_report: str = Field("", description="Raw LLM output")
    error: str | None = Field(None, description="Error message if prediction failed")


class _PeptideOutputSchema:
    """Backward compatibility class for legacy code."""

    schema = (
        "PH: (a,b) or (a,b] or [a,b) or [a,b]\n"
        "Concentration (log M): (a,b) or (a,b] or [a,b) or [a,b]\n"
        "Temperature (C): (a,b) or (a,b] or [a,b) or [a,b]\n"
        "Solvent: <single word or phrase>\n"
        "Estimated Time (minutes): (a,b) or (a,b] or [a,b) or [a,b]"
    )


peptide_output_schema = _PeptideOutputSchema()
