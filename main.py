from landslide_ai.pipeline.orchestrator import LandslideSystemOrchestrator


def main() -> None:
    orchestrator = LandslideSystemOrchestrator()
    result = orchestrator.run_demo()

    print("Landslide Risk Assessment Demo")
    print("=" * 36)
    print(f"Region: {result.district}, {result.state}")
    print(f"Risk level: {result.risk_level}")
    print(f"Risk score: {result.risk_score:.2f}")
    print(f"Predicted next 12h risk: {result.forecast_risk:.2f}")
    print(f"Advisory: {result.advisory}")


if __name__ == "__main__":
    main()
