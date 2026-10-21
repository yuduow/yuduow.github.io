# Article Figure Presentation

This edition uses Outfit, a consistent gray/blue/purple/teal method palette, and slightly larger labels. `figures.py` derives from the archived article figure generator. Only typography, colors, spacing and the tile-lifetime variant label are refined.

Run `python presentation/render.py` from the article directory with Matplotlib and NumPy available. The script verifies its archived inputs against the evidence bundle manifest, checks the 32 completion session medians and kernel launch counts, then regenerates all 24 desktop/mobile SVG/PNG files. It requires the resulting completion values, paired retention values, aggregation and axes to equal the archived figure provenance exactly.

The article renderer selects these presentation assets when present. The archived evidence and original figure assets remain byte-for-byte intact. `provenance.json` identifies the inputs, scripts, outputs and text-layout checks for this edition.

Layout diagnostics use full font metric rectangles, including Outfit’s generous leading. Those rectangles can overlap even when painted glyphs do not. Review diagnostics in the rendered SVG/PNG; the article edition received a visual check of all 12 layouts.
