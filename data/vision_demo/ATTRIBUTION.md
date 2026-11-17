# Demo images: attribution

The images in `frames/` are taken from the **BeltCrack** dataset by Huang, Ji, Ma and Ye (*Pattern Recognition*,
2026, doi:10.1016/j.patcog.2026.113598; data: doi:10.57760/sciencedb.31181), licensed
**CC BY-SA 4.0** (https://creativecommons.org/licenses/by-sa/4.0/). They come from the BeltCrack9kd test split.

Changes: each frame was reduced from 1024 x 576 (itself reduced from the original video resolution) to 640 x 360
and re-encoded as JPEG. Each clean image is a region of a test frame chosen to avoid every labelled crack, reduced
by the same factor and centred on a grey (114) 640 x 360 canvas.
File names are hashes. `catalog.json` lists the source frame (and crop) of every file. These images and this
folder are shared under the same licence, CC BY-SA 4.0.

Built by `python -m simulator.vision build-demo`. The demo scenario's captures (simulator/vision.py) refer to them.
