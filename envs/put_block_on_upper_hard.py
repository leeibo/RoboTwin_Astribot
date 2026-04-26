from .put_block_on import put_block_on


class put_block_on_upper_hard(put_block_on):
    PLATE_LAYER = "upper"
    BLOCK_COUNT = 3
    BLOCK_LAYER_SEQUENCE = ("lower", "lower", "upper")
