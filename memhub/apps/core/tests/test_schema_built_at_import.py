"""Model schemas must be built at import time, not deferred to runtime on the event loop.

pydantic leaves core schemas unbuilt when it meets forward references, generating them on first use.
That "first use" usually lands on the first request, i.e. on the event loop, and generation walks
inspect.signature → tokenize; the watchdog dump at core.log line 1973 on the real machine shows
MainThread stuck right on that path.

So this guard does not look at elapsed time (it varies with the machine), only at one \
deterministic fact:
after imports finish, is any model schema still unbuilt.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

import pydantic

import one_memory


def _every_model() -> list[type[pydantic.BaseModel]]:
    unimportable: list[str] = []
    for info in pkgutil.walk_packages(one_memory.__path__, "one_memory."):
        try:
            importlib.import_module(info.name)
        except ImportError:
            # A missing optional dependency is a known case. Every other failure must be reported:
            # silently skipping a module would hide exactly the incomplete model inside it.
            unimportable.append(info.name)
    assert not unimportable, (
        f"these modules failed to import, their models went unchecked: {sorted(unimportable)}"
    )
    seen: dict[str, type[pydantic.BaseModel]] = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith("one_memory"):
            continue
        for attribute in vars(module).values():
            if (
                isinstance(attribute, type)
                and issubclass(attribute, pydantic.BaseModel)
                and attribute.__module__.startswith("one_memory")
            ):
                seen[f"{attribute.__module__}.{attribute.__qualname__}"] = attribute
    return list(seen.values())


def test_every_model_has_its_schema_built_before_the_first_request():
    models = _every_model()
    # First prove this ruler actually measures something, or "no incomplete models" may just mean
    # no models were collected at all -- which would keep this guard forever green.
    assert len(models) > 100, (
        f"collected only {len(models)} models, the guard is not measuring anything real"
    )

    deferred = sorted(
        f"{model.__module__}.{model.__qualname__}"
        for model in models
        if not model.__pydantic_complete__
    )
    assert not deferred, (
        "these model schemas wait until first use to generate, and that call likely lands on the "
        "event loop: "
        f"{deferred}. Call model_rebuild() once the class the forward reference points to is "
        "defined."
    )
