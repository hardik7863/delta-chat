# pair1 — native/native, controlled edits

- **Rev A**: `revA.pdf` — verbatim copy of the provided *Lift Gas compressor* P&ID (AutoCAD Plant 3D plot, born-digital).
- **Rev B**: `revB.pdf` — Rev A with 6 authored edits applied via PyMuPDF (`src/synth.py`).
- **Ground truth**: `expected_delta.json` — the 6 edits, labelled, used by `make eval`.

Edits: line size 10"→8" (VF-43-9025); line spec AC21S→GC11S (VF-43-9029); setpoint HH 245→300; valve 43BL9070 removed; valve 26BL9073 moved +34pt; instrument PT-9099 added.
