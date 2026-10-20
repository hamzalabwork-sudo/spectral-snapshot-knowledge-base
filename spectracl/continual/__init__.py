from spectracl.continual.ewc import EWC
from spectracl.continual.replay import ReplayBuffer
from spectracl.continual.si import SI
from spectracl.continual.mas import MAS
from spectracl.continual.derpp import DERPPBuffer
from spectracl.continual.spectral_replay import SpectralReplayBuffer
from spectracl.continual.spectral_regularization import SpectralRegularizer
from spectracl.continual.null_space import NullSpaceProjector
from spectracl.continual.state_partition import StatePartitioner
from spectracl.continual.hint_loss import FrequencyHintLoss
from spectracl.continual.spectral_gradient import SpectralGradientDecomposer
from spectracl.continual.adaptivecl import AdaptiveCLBuffer
from spectracl.continual.clser import CLSERAdapter, CLSERBuffer

__all__ = [
    "EWC", "ReplayBuffer", "SI", "MAS", "DERPPBuffer",
    "SpectralReplayBuffer", "SpectralRegularizer",
    "NullSpaceProjector", "StatePartitioner",
    "FrequencyHintLoss", "SpectralGradientDecomposer",
    "AdaptiveCLBuffer",
    "CLSERAdapter", "CLSERBuffer",
]
