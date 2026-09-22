from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from landslide_ai.models.schemas import RiskAssessment


@dataclass(slots=True)
class AlertRecord:
    region_id: str
    state: str
    district: str
    severity: str
    alert_band: str
    risk_score: float
    forecast_risk: float
    calibrated_risk_score: float
    confidence_level: str
    recommended_action: str
    issued_at: str
    message: str


def build_alerts(
    assessments: list[RiskAssessment],
    high_threshold: float = 0.75,
    moderate_threshold: float = 0.55,
) -> list[AlertRecord]:
    issued_at = datetime.now(timezone.utc).isoformat()
    alerts: list[AlertRecord] = []

    for assessment in assessments:
        severity = _severity_for_assessment(
            assessment=assessment,
            high_threshold=high_threshold,
            moderate_threshold=moderate_threshold,
        )
        if severity == "None":
            continue

        alerts.append(
            AlertRecord(
                region_id=assessment.region_id,
                state=assessment.state,
                district=assessment.district,
                severity=severity,
                alert_band=assessment.alert_band,
                risk_score=assessment.risk_score,
                forecast_risk=assessment.forecast_risk,
                calibrated_risk_score=assessment.calibrated_risk_score,
                confidence_level=assessment.confidence_level,
                recommended_action=assessment.recommended_action,
                issued_at=issued_at,
                message=_build_message(assessment, severity),
            )
        )

    return alerts


def _severity_for_assessment(
    assessment: RiskAssessment,
    high_threshold: float,
    moderate_threshold: float,
) -> str:
    if assessment.calibrated_risk_score >= high_threshold or assessment.alert_band == "Critical":
        return "High"
    if assessment.calibrated_risk_score >= moderate_threshold or assessment.alert_band in {"Watch", "Review"}:
        return "Moderate"
    return "None"


def _build_message(assessment: RiskAssessment, severity: str) -> str:
    return (
        f"{severity} alert for {assessment.district}, {assessment.state}. "
        f"Current risk is {assessment.risk_score:.2f}, forecast risk is {assessment.forecast_risk:.2f}, "
        f"and calibrated operational risk is {assessment.calibrated_risk_score:.2f} with {assessment.confidence_level.lower()} confidence. "
        f"Recommended action: {assessment.recommended_action}"
    )
