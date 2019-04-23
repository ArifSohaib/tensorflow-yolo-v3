_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

_ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]
def get_batch_norm_params(is_training):
    return {
            'decay': _BATCH_NORM_DECAY,
            'epsilon': _BATCH_NORM_EPSILON,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
    }