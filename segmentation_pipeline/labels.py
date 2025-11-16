import numpy as np

CLASS_NAMES = [
    "Building",
    "Land",
    "Road",
    "Vegetation",
    "Water",
    "Unlabeled",
]

COLOR_CODES = [
    "#3C1098",
    "#8429F6",
    "#6EC1E4",
    "#FEDD3A",
    "#E2A929",
    "#9B9B9B",
]

COLOR_MAP = np.array(
    [tuple(int(code.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) for code in COLOR_CODES],
    dtype=np.uint8,
)
