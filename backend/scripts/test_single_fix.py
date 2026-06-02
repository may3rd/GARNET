import sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from garnet.pid_extractor import PIDPipeline, PipelineConfig

OUTPUT_ROOT = Path("/Users/maetee/Documents/1. PROJECTS/claude-workspace/projects/gcme/1_active/garnet/output")
label = "ppcl"
image_path = Path("/Volumes/Ginnungagap/maetee/Code/GARNET/backend/test/ppcl/Test-00001.jpg")
out_dir = OUTPUT_ROOT / label / "default"
out_dir.mkdir(parents=True, exist_ok=True)

os.chdir(str(out_dir))

cfg = PipelineConfig(ocr_route="ocrmac")
pipe = PIDPipeline(str(image_path), out_dir="output", cfg=cfg)
pipe.run()
print("Done")
