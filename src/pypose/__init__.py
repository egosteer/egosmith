"""
Lightweight stub of the pypose API used by DPVO loop_closure.optim_utils.

We only need this to satisfy imports; in our config LOOP_CLOSURE and
CLASSIC_LOOP_CLOSURE are disabled, so these functions are never actually
executed in our pipeline.
"""
import torch


class Sim3:
    def __init__(self, data):
        self.data = torch.as_tensor(data, dtype=torch.float32)

    def Inv(self):
        # Return self for stub purposes; real logic is unused when loop closure is off.
        return self

    def Log(self):
        # Return an object with .tensor() to match optim_utils expectations.
        class _LogObj:
            def __init__(self, t):
                self._t = t

            def tensor(self):
                return self._t

        return _LogObj(self.data)


class SE3:
    def __init__(self, data):
        self.data = torch.as_tensor(data, dtype=torch.float32)

    def Inv(self):
        return self


def Exp(x):
    # Minimal Exp returning a Sim3 wrapper; not used in our non-loop-closure config.
    return Sim3(x)

