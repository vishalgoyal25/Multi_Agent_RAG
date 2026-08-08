"""Compatibility shim for a confirmed upstream bug in ragas==0.4.3.

`ragas/llms/base.py` unconditionally imports `ChatVertexAI` from
`langchain_community.chat_models.vertexai` — a path LangChain itself removed
when Vertex AI support was split out into the separate
`langchain-google-vertexai` package. The result: `import ragas.llms` fails
outright for every ragas 0.4.x user, regardless of which LLM provider they
actually use. Confirmed upstream, not fixed as of this writing:
    https://github.com/vibrantlabsai/ragas/issues/2741
    https://github.com/vibrantlabsai/ragas/issues/2745

This registers a fake `langchain_community.chat_models.vertexai` module in
`sys.modules`, pointing at the real `ChatVertexAI` from its current home, so
ragas's broken import resolves without patching, forking, or downgrading
ragas itself. Nothing in this project calls Vertex AI or needs Google Cloud
credentials — the shim exists solely to satisfy an eager import in code we
don't control.

Import order matters: this module must be imported *before* anything imports
from `ragas`, since Python checks `sys.modules` for the exact dotted path
before touching the filesystem — that's the mechanism this relies on.
"""

from __future__ import annotations

import sys
import types

from langchain_google_vertexai import ChatVertexAI

_shim = types.ModuleType("langchain_community.chat_models.vertexai")
_shim.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]

# setdefault, not direct assignment: if a future ragas release fixes the
# import and this shim is imported by mistake, a real module already present
# in sys.modules is left untouched rather than silently overridden.
sys.modules.setdefault("langchain_community.chat_models.vertexai", _shim)
