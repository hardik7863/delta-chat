# pair3 — synthetic DXF pair (DWG/DXF adapter)

- **Rev A / Rev B**: authored with `ezdxf` (`src/synth_dwg.py`) — TEXT tags + LINE pipes in a compact P&ID layout.
- **DWG note**: the adapter accepts `.dwg` and converts DWG→DXF via ODA File Converter when present; we ship `.dxf` directly so the pair runs with no external tool. The conversion seam is real (see `src/ingest/dwg.py`).
- **Ground truth**: 4 edits — size 10"→8", valve 43BL9070 removed, PT-9099 added, valve 40BL9021 moved.
