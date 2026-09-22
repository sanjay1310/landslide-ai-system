# Final Year Project Writeup Material

## Abstract

This project presents a software-only multi-agent AI system for real-time landslide detection, prediction, and advisory generation using weather, terrain, and remote sensing data. The system integrates rainfall-derived indicators, terrain attributes, spatial graph propagation, and explainable advisory generation into a unified platform. A forecasting module supports time-series rainfall prediction, while a dashboard and API expose risk maps, alerts, and scenario simulation for Indian regions. The project is designed as a deployable analytics platform rather than a hardware-dependent prototype, making it suitable for academic demonstration, extension, and publication-oriented research.

## Objectives

1. Build a software-only landslide early warning pipeline for Indian regions.
2. Fuse weather, terrain, and remote-sensing style features into a single risk framework.
3. Use a multi-agent architecture for modular analysis and explainability.
4. Model spatial dependency between nearby districts using graph propagation.
5. Provide human-readable advisories and interactive risk visualization.
6. Support deployable interfaces through dashboard and API layers.

## Problem Statement

Landslides are strongly influenced by rainfall intensity, terrain steepness, vegetation condition, and spatial spillover from nearby regions. Many student projects rely on isolated classification or hardware-centric sensing, which limits scalability and reproducibility. This project addresses the need for a software-only regional landslide intelligence platform that supports ingestion, prediction, forecasting, explainability, and deployment.

## Methodology

1. Collect region-wise weather and terrain data.
2. Generate fused district-level feature tables.
3. Build rainfall history windows for forecasting.
4. Compute vision-style, prediction, and graph signals.
5. Aggregate signals into risk and forecast scores.
6. Generate explainable advisories and alerts.
7. Serve outputs through dashboard and API.

## Innovation Points

- Multi-agent architecture
- Spatial graph propagation
- Scenario simulation
- Saved inference artifacts
- API-ready deployment design
- Explainable district advisories

## Results to Present

- District-wise risk table
- India heat map and district boundary view
- Alert center output
- Forecast metrics such as MAE, RMSE, and R2
- Scenario simulation showing risk increase under heavier rainfall

## Limitations

- Real external datasets still require deployment-time configuration
- Placeholder GIS boundaries should be replaced with official files
- The current spatial module is graph-based but not yet a trainable GNN
- The LSTM path requires PyTorch installation and richer history

## Future Scope

- Full Sentinel raster processing
- Trainable GNN for spatial dependency
- Real-time scheduled ingestion
- Mobile alert delivery
- Authenticated production deployment
