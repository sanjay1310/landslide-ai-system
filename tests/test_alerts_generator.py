from __future__ import annotations

import unittest

from landslide_ai.alerts.generator import build_alerts
from landslide_ai.models.schemas import RiskAssessment


class AlertsGeneratorTest(unittest.TestCase):
    def test_build_alerts_uses_calibrated_risk_and_confidence_fields(self) -> None:
        assessments = [
            RiskAssessment(
                region_id="R1",
                state="Kerala",
                district="Idukki",
                latitude=10.1,
                longitude=76.9,
                risk_score=0.81,
                forecast_risk=0.84,
                risk_level="High",
                primary_driver="Rainfall",
                uncertainty_score=0.05,
                graph_exposure=0.34,
                propagated_graph_exposure=0.42,
                rainfall_exposure=0.86,
                terrain_exposure=0.61,
                vegetation_exposure=0.31,
                calibrated_risk_score=0.79,
                confidence_level="High",
                alert_band="Critical",
                recommended_action="Escalate district monitoring, verify slope conditions, and prepare field advisories.",
            )
        ]

        alerts = build_alerts(assessments)

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, "High")
        self.assertEqual(alerts[0].alert_band, "Critical")
        self.assertEqual(alerts[0].confidence_level, "High")
        self.assertIn("calibrated operational risk", alerts[0].message)


if __name__ == "__main__":
    unittest.main()
