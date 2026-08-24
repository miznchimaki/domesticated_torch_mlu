import os
from typing_extensions import override

from ...utils import gorilla
from torch._inductor.runtime.autotune_cache import _LocalAutotuneCacheBackend


@override
def _put(self, key: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(key), exist_ok=True)
    from torch._inductor import codecache

    codecache.write_atomic(key, data)


patch = gorilla.Patch(_LocalAutotuneCacheBackend, "_put", _put)
gorilla.apply(patch)
