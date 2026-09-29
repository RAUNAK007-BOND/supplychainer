# Supplychainer — public demo

Plan the fastest, best-value and most reliable way to ship between 444 ports, airports and
logistics hubs. Try a sample trip, simulate a disruption such as the Suez Canal blockage, and
watch the route change around it.

This repository is the deployable build of the Supplychainer web app (FastAPI backend plus the
pre-built React interface), configured for Render's free plan by `render.yaml`.

- Delay predictions come from quantile machine-learning models (typical, planned and worst-case arrival).
- Disruption scenarios reroute cargo around blocked canals, straits and ports.
- Watch a route, turn on a disruption, and get a live alert with a suggested new plan.

On this free public demo, everyone shares the same saved routes and alerts, data resets when the
service restarts, webhooks are turned off, and the live news check uses keyword scoring instead of
the full language model.
