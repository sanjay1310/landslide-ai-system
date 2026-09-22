from landslide_ai.models.schemas import AgentSignals, RiskAssessment
from landslide_ai.services.rag_service import get_rag_service


class AdvisoryGenerationAgent:
    """Converts model signals into a human-readable advisory."""

    def __init__(self, project_root: str = ".") -> None:
        self.rag = get_rag_service(project_root)

    def generate(self, signals: AgentSignals, assessment: RiskAssessment) -> str:
        signal_map = {
            "rainfall intensity": signals.rainfall_signal,
            "terrain steepness": signals.terrain_signal,
            "vegetation stress": signals.vegetation_signal,
            "regional spillover": signals.graph_signal,
            "propagated network spillover": signals.propagated_graph_signal,
        }
        ordered_drivers = sorted(signal_map.items(), key=lambda item: item[1], reverse=True)
        primary_driver = ordered_drivers[0][0]
        secondary_driver = ordered_drivers[1][0]

        severity_line = {
            "High": (
                f"High landslide risk is active in {assessment.district}, {assessment.state}. "
                f"Current risk is {assessment.risk_score:.2f} and forecast risk for the next 12 hours is {assessment.forecast_risk:.2f}."
            ),
            "Moderate": (
                f"Moderate landslide risk is present in {assessment.district}, {assessment.state}. "
                f"Current risk is {assessment.risk_score:.2f} with forecast risk rising to {assessment.forecast_risk:.2f}."
            ),
            "Low": (
                f"Low but non-zero landslide risk is estimated for {assessment.district}, {assessment.state}. "
                f"Current risk is {assessment.risk_score:.2f} and forecast risk is {assessment.forecast_risk:.2f}."
            ),
        }[assessment.risk_level]

        if assessment.risk_level == "High":
            action_line = (
                "Recommended action: intensify field monitoring, prepare district alert dissemination, "
                "check slope drainage conditions, and keep response teams on standby."
            )
        elif assessment.risk_level == "Moderate":
            action_line = (
                "Recommended action: maintain enhanced monitoring, review recent rainfall accumulation, "
                "and notify local authorities in vulnerable slope zones."
            )
        else:
            action_line = (
                "Recommended action: continue routine monitoring and watch for rapid rainfall escalation "
                "or new terrain instability signals."
            )

        network_line = (
            "Spatial network influence is elevated."
            if assessment.propagated_graph_exposure >= 0.60
            else "Spatial spillover remains limited."
        )
        uncertainty_line = (
            "Confidence is strong."
            if assessment.uncertainty_score <= 0.30
            else "Confidence is moderate because drivers are mixed."
            if assessment.uncertainty_score <= 0.60
            else "Confidence is lower and this district should be reviewed with supporting observations."
        )

        rag_query = (
            f"{assessment.risk_level} landslide guidance for {assessment.state} "
            f"primary driver {primary_driver} forecast risk {assessment.forecast_risk:.2f}"
        )
        retrieved_context = self.rag.summarize_for_advisory(rag_query, top_k=2)
        retrieved_line = f"Retrieved guidance: {retrieved_context}" if retrieved_context else ""

        return (
            f"{severity_line} "
            f"Primary driver: {primary_driver}; secondary driver: {secondary_driver}. "
            f"{network_line} "
            f"Model uncertainty is {assessment.uncertainty_score:.2f}. {uncertainty_line} "
            f"{action_line} "
            f"{retrieved_line}"
        )
