"""Compatibility only for the pinned NanoDet v1 checkout under PyTorch 2.x.

Python imports this module automatically when ``src`` is on ``PYTHONPATH``.
NanoDet v1 imports ``torch._six.string_classes`` in its collate implementation;
PyTorch removed that private module. Keep this narrow shim outside the upstream
checkout so its fixed commit remains inspectable and unmodified.
"""

from __future__ import annotations

import collections.abc
import sys
from types import ModuleType

try:
    import torch
except ImportError:
    torch = None

if torch is not None and "torch._six" not in sys.modules:
    torch_six = ModuleType("torch._six")
    torch_six.string_classes = (str, bytes)
    torch_six.int_classes = (int,)
    torch_six.container_abcs = collections.abc
    sys.modules["torch._six"] = torch_six
    torch._six = torch_six
