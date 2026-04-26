from .put_block_on import put_block_on


class put_block_on_upper_easy(put_block_on):
    PLATE_LAYER = "upper"
    BLOCK_COUNT = 2
    BLOCK_LAYER_SEQUENCE = ("lower", "lower")
