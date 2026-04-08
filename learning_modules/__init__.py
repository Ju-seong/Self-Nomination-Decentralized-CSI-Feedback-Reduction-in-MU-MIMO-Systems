"""
Learning modules for RL-based user scheduling and beamforming.
"""
from .reinforce_full_input import Modules as REINFORCE_FullInput
from .reinforce_channel_gain_input import Modules_chg as REINFORCE_ChannelGainInput
from .direct_gradient_full_input import Modules as DirectGradient_FullInput
from .direct_gradient_channel_gain_input import Modules as DirectGradient_ChannelGainInput

__all__ = [
    "REINFORCE_FullInput",
    "REINFORCE_ChannelGainInput",
    "DirectGradient_FullInput",
    "DirectGradient_ChannelGainInput",
]

MODULE_MAPPING = {
    ("reinforce", "full"): REINFORCE_FullInput,
    ("reinforce", "chg_input"): REINFORCE_ChannelGainInput,
    ("directgrad", "full"): DirectGradient_FullInput,
    ("directgrad", "chg_input"): DirectGradient_ChannelGainInput,
}


def get_module_class(method, input_type):
    """Return the module class for the given method and input type."""
    if (method, input_type) not in MODULE_MAPPING:
        raise ValueError(
            f"Invalid (method, input_type)=({method!r}, {input_type!r}). "
            f"Available: {list(MODULE_MAPPING.keys())}"
        )
    return MODULE_MAPPING[(method, input_type)]
