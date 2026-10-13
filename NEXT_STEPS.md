# NEXT STEPS (handoff for Antigravity / any coding agent)

## What exists and was tested (v0.1)
Verified by running the server against a synthetic belt video + the simulator:
stream connect/reconnect, MJPEG feed with overlay, belt edge/misalignment, dust %, blur check, sensor ingest
(HTTP), simulator + fault injection, health score, alert engine (dedupe, rate-limit, snapshots, ack, CSV),
anomaly detector + RF risk/RUL (synthetic bootstrap), SQLite storage, WebSocket 1 Hz snapshot,
dataset capture, settings persistence. Frontend JS is syntax-checked but was NOT exercised in a browser.

## Written but NOT run (needs ultralytics/torch + real data)
* `backend/training/trainer.py` -> `TrainingManager` (YOLO training thread, versioning, metrics) and zip import
* `backend/vision/detector.py` -> loading custom weights, inference
* `backend/sensors/mqtt_client.py` -> MQTT bridge
* Frontend pages for training / dataset / model deploy

## Not built yet (priority order)
1. **In-browser labelling tool** (canvas bounding boxes -> `POST /api/labels/{image}`; endpoints exist, use
   `CLASSES` from `/api/settings`) + auto-label assist (run current model, accept/correct boxes).
2. **ROI picker**: draw the belt region on a snapshot instead of typing numbers (settings.roi = [x,y,w,h] normalised).
3. **Joint monitoring**: mark joint position; track defect count there separately; per-joint alerts.
4. **Digital-twin view**: schematic belt showing where along its length faults occurred (needs belt-position
   estimate from speed x time, or optical-flow odometry).
5. **Anomaly-detection alternative** for users without defect images (PatchCore/autoencoder on healthy frames).
6. **Real predictor training**: replace the synthetic bootstrap with labelled run-to-failure logs from
   `feature_rows`; add a "mark failure/maintenance event" UI to create labels.
7. Alert hooks (webhook / email / Telegram), snapshot retention cleanup, auth for the API, Docker + compose,
   pytest suite (health, alerts dedupe, belt analysis on generated frames).

## Conventions
* Settings live in `backend/config.py::Settings` (persisted to data/settings.json); thresholds are never hard-coded elsewhere.
* Vision runs in threads in `video.py`; fusion/alerts/prediction in the async loop in `monitor.py` (1 Hz).
* Keep the WebSocket payload shape in `Monitor.snapshot()` and `frontend/js/app.js::handleLive` in sync.
* Missing sensors/camera/model must degrade gracefully (health re-normalises over available data).

## Paste this to the agent
> Read README.md and NEXT_STEPS.md, run the project, fix anything broken, then implement items 1 and 2 from
> "Not built yet" (labelling tool, ROI picker). Test against `scripts/make_test_video.py`. Do not change the API
> shapes the frontend already uses without updating both sides.
