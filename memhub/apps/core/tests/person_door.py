"""Writing memories as a person in tests: claim the person door once per governance store, then \
reuse it."""

import weakref

from one_memory.memory_governance import MemoryGovernanceStore, PersonWrites

_doors: weakref.WeakKeyDictionary[MemoryGovernanceStore, PersonWrites] = weakref.WeakKeyDictionary()


def as_person(store: MemoryGovernanceStore | PersonWrites) -> PersonWrites:
    """A store Core opened has already handed its door over: pass `app.state.person_writes`."""
    if isinstance(store, PersonWrites):
        return store
    door = _doors.get(store)
    if door is None:
        door = _doors[store] = store.claim_person_writes("user")
    return door
