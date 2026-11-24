"""Deterministic policy over matured chronological monitoring windows."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Policy:
    min_samples: int = 20
    min_coverage: float = 0.8
    consecutive_weeks: int = 2
    excess_error: float = 0.15
    drift_z: float = 1.5
    validation_weeks: int = 8
    cooldown_hours: float = 24
    daily_budget: int = 2
    max_lag_days: int = 21

    def __post_init__(self) -> None:
        if not (
            1 <= self.consecutive_weeks <= 8
            and self.min_samples >= 1
            and 0 < self.min_coverage <= 1
            and self.daily_budget >= 1
            and self.max_lag_days >= 0
            and 1 <= self.validation_weeks <= 26
            and math.isfinite(self.excess_error)
            and self.excess_error > -1
        ):
            raise ValueError("Invalid policy sample, coverage, freshness or budget limit")
        for value in (self.cooldown_hours, self.drift_z):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Policy thresholds must be finite and nonnegative")


def decide(
    policy: Policy,
    windows: list[dict],
    reference: float | None,
    *,
    quality: str | None,
    drift: bool,
    busy: bool,
    cooldown: bool,
    spent: bool,
    trainable: bool,
) -> tuple[str, str]:
    if quality:
        return "QUARANTINE", quality
    if reference is None or not math.isfinite(reference):
        return "INVESTIGATE", "A finite development error reference is required"
    recent = windows[-policy.consecutive_weeks :]
    if len(recent) < policy.consecutive_weeks or any(
        w["samples"] < policy.min_samples
        or w["coverage"] < policy.min_coverage
        or w["seasonal_scaled_error"] is None
        for w in recent
    ):
        return "WAIT_FOR_LABELS", "Too few mature, sufficiently covered weekly windows"
    if any(w["seasonal_scaled_error"] <= reference * (1 + policy.excess_error) for w in recent):
        if drift:
            return "INVESTIGATE", "Input volumes shifted without persistent error degradation"
        return "HOLD", "Weekly errors are within the development tolerance"
    if busy or cooldown or spent:
        return "HOLD", "An active job, cooldown or daily training budget prevents retraining"
    if not trainable:
        return "WAIT_FOR_LABELS", "Ingest complete labeled weeks before fitting a challenger"
    return "RETRAIN", "Persistent seasonally scaled error exceeds the development tolerance"
