from .put_block_on import put_block_on


class put_block_on_lower_easy(put_block_on):
    PLATE_LAYER = "lower"
    BLOCK_COUNT = 2
    BLOCK_LAYER_SEQUENCE = ("upper", "upper")
    PLATE_PLACE_SLOT_OFFSETS = {
        1: ((0.0, 0.0),),
        2: ((0, 0.04), (0, -0.04)),
        3: ((0.045, 0.0), (-0.023, 0.040), (-0.023, -0.040)),
    }

