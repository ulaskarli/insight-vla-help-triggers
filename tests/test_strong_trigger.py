import numpy as np

from insight.strong_trigger import StrongHelpTrigger


def test_feature_order_and_trim_without_loading_checkpoint():
    trigger = StrongHelpTrigger.__new__(StrongHelpTrigger)
    trigger.trim_head = 3
    trigger.trim_tail = 2

    output = {
        "au": np.arange(10),
        "eu": np.arange(10) + 100,
        "entropy": np.arange(10) + 200,
        "perplexity": np.arange(10) + 300,
    }
    x = trigger.features_from_policy_output(output)
    assert x.shape == (5, 4)
    np.testing.assert_array_equal(x[:, 0], np.arange(3, 8))
    np.testing.assert_array_equal(x[:, 1], np.arange(103, 108))
    np.testing.assert_array_equal(x[:, 2], np.arange(203, 208))
    np.testing.assert_array_equal(x[:, 3], np.arange(303, 308))
