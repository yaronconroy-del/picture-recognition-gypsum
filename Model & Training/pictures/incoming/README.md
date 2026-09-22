# incoming/

Watched folder for `scripts/watch_camera_feed.py` — the simulated live-feed
demo. Drop a `.jpg`/`.jpeg`/`.png` file in here while the script is running
to simulate the camera producing a new frame; it gets classified within a
few seconds and an alert is printed if the prediction is INVALID.

Empty by design — this is a drop point for the demo, not part of the
training dataset.
