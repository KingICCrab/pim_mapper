
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve() / "src"))
sys.path.insert(0, str(Path(".").resolve()))

from nn_dataflow.nns import import_network

network = import_network("resnet50")
for layer_name in network:
    print(layer_name)
