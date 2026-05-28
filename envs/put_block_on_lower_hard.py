from .put_block_on import put_block_on


class put_block_on_lower_hard(put_block_on):
    PLATE_LAYER = "lower"
    BLOCK_COUNT = 3
    BLOCK_LAYER_SEQUENCE = ("upper", "upper", "lower")
    PLATE_PLACE_SLOT_OFFSETS = {
        1: ((0.0, 0.0),),
        2: ((0, 0.04), (0, -0.04)),
        3: ((0, 0.04), (0, -0.04), (0.04, 0)),
    }
    LOWER_PLACE_DIS = 0.05
    BLOCK_SIZE_RANGE = (0.015, 0.02)
