"""Interface for poison recipes."""
from .witch_matching import WitchGradientMatching
# from .witch_metapoison import WitchMetaPoison, WitchMetaPoisonHigher, WitchMetaPoison_v3
# from .witch_watermark import WitchWatermark
# from .witch_poison_frogs import WitchFrogs
# from .witch_bullseye import WitchBullsEye
# from .witch_patch import WitchPatch
# from .witch_htbd import WitchHTBD
# from .witch_convex_polytope import WitchConvexPolytope

import torch


def Witch(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args.recipe == 'gradient-matching':
        return WitchGradientMatching(args, setup)
    else:
        raise NotImplementedError()


__all__ = ['Witch']
