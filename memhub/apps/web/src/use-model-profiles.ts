import type {
  ModelCreationOperation,
  ModelProbeResult,
  ModelProfile,
  ModelProfileInput,
  ModelProfilePatch,
} from "@one-memory/contracts";
import axios from "axios";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  abandonModelCreationOperation,
  activateModelProfile,
  apiErrorCode,
  createModelProfile,
  deleteModelProfile,
  getModelCreationOperation,
  getModelProfiles,
  probeModelProfile,
  updateModelProfile,
} from "./api";
import { modelErrorText } from "./model-error-text";
import {
  clearPendingModelCreation,
  readPendingModelCreation,
  savePendingModelCreation,
} from "./pending-model-creation";
import {
  type PendingModelProbe,
  readPendingModelProbe,
} from "./pending-model-probe";
import { traceUiAction, type UiTraceAction } from "./ui-trace";
import { isDefiniteRefusal } from "./write-outcome";

/** Map a write's internal key to the closed action vocabulary. `create`/
 * `activate`/`update`/`delete` are submits; recovery and probe re-read state. */
function traceAction(key: string): UiTraceAction {
  if (key.startsWith("probe:")) return "retry";
  if (key === "creation-recovery") return "retry";
  return "submit";
}

/** A stable, bounded reason code for the log — never the raw key.
 *
 * `key` carries the op and the target (`activate:<profile-id>`), so using it
 * verbatim as `reason` would put a profile UUID in every line and make each
 * failure its own value, defeating the point of a queryable log. Map it to a
 * small closed set instead.
 */
function traceReason(key: string): string {
  if (key === "create") return "model_create_failed";
  if (key.startsWith("probe:")) return "model_probe_failed";
  if (key.startsWith("activate:")) return "model_activate_failed";
  if (key.startsWith("update:")) return "model_update_failed";
  if (key.startsWith("delete:")) return "model_delete_failed";
  if (key === "creation-recovery") return "model_creation_recovery_failed";
  return "model_operation_failed";
}

/** The same bounded reasons, marked as *unknown* rather than failed. */
function traceReasonUnknown(key: string): string {
  const failed = traceReason(key);
  return failed.replace(/_failed$/, "_result_unknown");
}

export function useModelProfiles() {
  const [profiles, setProfileState] = useState<ModelProfile[]>([]);
  const profilesRef = useRef<ModelProfile[]>([]);
  // Operation receipts must be usable by the next step before React renders.
  const setProfiles = useCallback(
    (next: ModelProfile[] | ((previous: ModelProfile[]) => ModelProfile[])) => {
      profilesRef.current =
        typeof next === "function" ? next(profilesRef.current) : next;
      setProfileState(profilesRef.current);
    },
    [],
  );
  const [probe, setProbe] = useState<ModelProbeResult | null>(null);
  const [probeResults, setProbeResults] = useState<
    Record<string, ModelProbeResult>
  >({});
  const [probeErrors, setProbeErrors] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [pendingCreation, setPendingCreation] = useState<string | null>(null);
  const [creationMissing, setCreationMissingState] = useState<string | null>(
    null,
  );
  const [creationStorageFailed, setCreationStorageFailed] = useState(false);
  const [pendingProbe, setPendingProbe] = useState<PendingModelProbe | null>(
    null,
  );
  const [probeStorageFailed, setProbeStorageFailed] = useState(false);
  const readProbePending = useCallback(() => {
    try {
      const value = readPendingModelProbe();
      setPendingProbe(value);
      setProbeStorageFailed(false);
      return value;
    } catch {
      setProbeStorageFailed(true);
      throw new Error("pending_model_probe_failed");
    }
  }, []);
  const pendingCreationRef = useRef<string | null>(null);
  const creationMissingRef = useRef<string | null>(null);
  const alive = useRef(false);
  const session = useRef(0);
  const readVersion = useRef(0);
  const flight = useRef<{ key: string; promise: Promise<unknown> } | null>(
    null,
  );
  const setCreationMissing = useCallback((value: string | null) => {
    creationMissingRef.current = value;
    setCreationMissingState(value);
  }, []);

  const refresh = useCallback(
    async (afterOperation = false) => {
      const version = ++readVersion.current;
      const owner = session.current;
      try {
        const next = await getModelProfiles();
        if (
          !alive.current ||
          owner !== session.current ||
          version !== readVersion.current
        )
          return;
        setProfiles(next);
        setRefreshError(null);
      } catch {
        if (
          !alive.current ||
          owner !== session.current ||
          version !== readVersion.current
        )
          return;
        setRefreshError(
          afterOperation
            ? "The operation completed, but the configuration list was not refreshed. The operation receipt is kept; refresh and verify, and do not save again."
            : "The model configuration cannot be refreshed right now; confirmed configurations are kept. Please retry the refresh.",
        );
      }
    },
    [setProfiles],
  );

  useEffect(() => {
    alive.current = true;
    try {
      const pending = readPendingModelCreation();
      pendingCreationRef.current = pending;
      setPendingCreation(pending);
    } catch {
      setCreationStorageFailed(true);
    }
    try {
      readProbePending();
    } catch {
      /* Storage failure is surfaced separately. */
    }
    void refresh();
    return () => {
      alive.current = false;
      session.current++;
      readVersion.current++;
      flight.current = null;
      creationMissingRef.current = null;
    };
  }, [refresh, readProbePending]);

  const run = useCallback(
    <T>(
      key: string,
      operation: (traceId: string) => Promise<T>,
      commit: (result: T) => void,
      onFailure?: (reason: unknown) => void,
    ): Promise<T | undefined> => {
      if (!alive.current) return Promise.resolve(undefined);
      try {
        const pending = readProbePending();
        if (pending && key !== `probe:${pending.profileId}`) {
          setError(
            "Please verify the last model test result first; the configuration must not be changed or activated.",
          );
          return Promise.resolve(undefined);
        }
      } catch {
        return Promise.resolve(undefined);
      }
      if (
        key !== "creation-recovery" &&
        (pendingCreationRef.current || creationStorageFailed)
      )
        return Promise.resolve(undefined);
      if (flight.current)
        return flight.current.key === key
          ? (flight.current.promise as Promise<T | undefined>)
          : Promise.resolve(undefined);
      const owner = session.current;
      readVersion.current++;
      setBusy(key);
      setError(null);
      setRefreshError(null);
      const promise = (async () => {
        // One choke point for every model-profile write, so each gets a
        // ui.action/ui.result pair whose trace id reaches Core. `key` already
        // names the operation, so it doubles as the trace's reason on failure.
        const trace = traceUiAction("settings", traceAction(key));
        try {
          const result = await Promise.resolve().then(() =>
            operation(trace.traceId),
          );
          if (!alive.current || owner !== session.current) return undefined;
          trace.done("succeeded");
          commit(result);
          // A successful write is authoritative even if its follow-up GET fails.
          void refresh(true);
          return result;
        } catch (reason) {
          // Only a definite refusal is a failure. A timeout or a 5xx leaves the
          // write's outcome unknown — for a create that is exactly the case the
          // pending-creation journal exists to reconcile, so the log must not
          // claim a failure the UI cannot prove.
          trace.done(isDefiniteRefusal(reason) ? "failed" : "unknown", {
            reason: isDefiniteRefusal(reason)
              ? traceReason(key)
              : traceReasonUnknown(key),
          });
          if (alive.current && owner === session.current) {
            if (onFailure) onFailure(reason);
            else setError(message(reason));
          }
          return undefined;
        } finally {
          if (alive.current && owner === session.current) {
            try {
              readProbePending();
            } catch {
              /* Keep the storage fence visible. */
            }
            flight.current = null;
            setBusy(null);
          }
        }
      })();
      flight.current = { key, promise };
      return promise;
    },
    [refresh, creationStorageFailed, readProbePending],
  );

  const mergeProfile = (profile: ModelProfile) =>
    setProfiles((previous) => {
      const others = previous.filter(
        (item) => item.profile_id !== profile.profile_id,
      );
      return [
        ...others.map((item) =>
          profile.is_active ? { ...item, is_active: false } : item,
        ),
        profile,
      ];
    });

  const probeKey = (value: Pick<ModelProfile, "profile_id" | "revision">) =>
    JSON.stringify([value.profile_id, value.revision]);

  function testProfile(profileId: string) {
    const current = profilesRef.current.find(
      (item) => item.profile_id === profileId,
    );
    const recovering =
      pendingProbe?.profileId === profileId ? pendingProbe : null;
    const inputRevision = current?.revision ?? recovering?.inputRevision;
    if (inputRevision === undefined) return Promise.resolve(undefined);
    const key = probeKey({ profile_id: profileId, revision: inputRevision });
    return run(
      `probe:${profileId}`,
      (traceId) => {
        setProbeResults((previous) => {
          const next = { ...previous };
          delete next[key];
          return next;
        });
        setProbeErrors((previous) => {
          const next = { ...previous };
          delete next[key];
          return next;
        });
        return probeModelProfile(profileId, inputRevision, traceId);
      },
      (result) => {
        if (
          result.profile_id !== profileId ||
          !Number.isSafeInteger(result.profile_revision) ||
          result.profile_revision < inputRevision
        )
          throw new Error("model_profile_probe_mismatch");
        setProbe(result);
        setProbeResults((previous) => ({
          ...previous,
          [probeKey({
            profile_id: profileId,
            revision: result.profile_revision,
          })]: result,
        }));
        setProfiles((previous) =>
          previous.map((item) =>
            item.profile_id === result.profile_id &&
            item.revision === result.profile_revision
              ? { ...item, capabilities: result.capabilities }
              : item,
          ),
        );
      },
      (reason) =>
        setProbeErrors((previous) => ({ ...previous, [key]: message(reason) })),
    );
  }

  return {
    profiles,
    probe:
      probe &&
      profiles.some(
        (item) =>
          item.profile_id === probe.profile_id &&
          item.revision === probe.profile_revision,
      )
        ? probe
        : null,
    probeFor: (profile: ModelProfile) =>
      probeResults[probeKey(profile)] ?? null,
    probeErrorFor: (profile: ModelProfile) =>
      probeErrors[probeKey(profile)] ?? null,
    busy,
    error,
    refreshError,
    pendingCreation,
    creationMissing,
    pendingProbe,
    probeStorageFailed,
    creationStorageFailed,
    reconcileCreation: () => {
      setCreationMissing(null);
      return run(
        "creation-recovery",
        async (traceId) => {
          // Recovery is a GET that re-reads a previous write, so the trace id
          // is unused: Core writes no kernel line for a GET.
          void traceId;
          const owner = session.current;
          const operationId = readPendingModelCreation();
          if (!operationId) {
            if (pendingCreationRef.current)
              throw Error("creation_record_missing");
            setCreationStorageFailed(false);
            return null;
          }
          const receipt = await getModelCreationOperation(operationId);
          if (!alive.current || owner !== session.current)
            throw Error("creation_owner_changed");
          if (
            receipt.operation_id === operationId &&
            receipt.state === "not_found" &&
            receipt.profile_id === null &&
            receipt.created_revision === null
          ) {
            setCreationMissing(operationId);
            throw Error("creation_receipt_not_found");
          }
          // `not_found` must NOT clear the pending record. Core says so
          // explicitly ("pending/not_found never authorize resending a POST",
          // `model_profiles.py:lookup_creation_operation`): the POST may still
          // be in flight, or have landed before its evidence row was written.
          // Clearing the lock here would let the user start a second create and
          // end up with two profiles. The record is released only on a terminal
          // receipt (committed/deleted) or an explicit user resolution.
          if (
            receipt.operation_id !== operationId ||
            !receipt.profile_id ||
            receipt.created_revision !== 1
          )
            throw Error("creation_receipt_unconfirmed");
          if (receipt.state === "deleted") {
            clearPendingModelCreation(operationId);
            return null;
          }
          if (receipt.state !== "committed")
            throw Error("creation_receipt_unconfirmed");
          const profile = (await getModelProfiles()).find(
            (item) =>
              item.profile_id === receipt.profile_id && item.revision >= 1,
          );
          if (!alive.current || owner !== session.current)
            throw Error("creation_owner_changed");
          if (!profile) throw Error("creation_profile_unavailable");
          clearPendingModelCreation(operationId);
          return profile;
        },
        (profile) => {
          pendingCreationRef.current = null;
          setPendingCreation(null);
          setCreationStorageFailed(false);
          setCreationMissing(null);
          if (profile) mergeProfile(profile);
        },
        (reason) =>
          setError(
            reason instanceof Error &&
              reason.message === "creation_receipt_not_found"
              ? "The previous creation record was not found. After you confirm dropping the old record you can reconfigure; the system will not retry automatically."
              : "The creation result is still pending verification; it will not be created again. Please check again shortly.",
          ),
      );
    },
    discardMissingCreation: async (): Promise<boolean | undefined> => {
      const operationId = pendingCreationRef.current;
      if (!alive.current || !operationId || flight.current) return undefined;
      const owner = session.current;
      const currentPending = () => {
        const current = readPendingModelCreation();
        if (current !== operationId) throw Error("creation_owner_changed");
        return current;
      };
      try {
        currentPending();
      } catch {
        setCreationStorageFailed(true);
        setCreationMissing(null);
        setError(
          "The local recovery record has changed; the lock is kept. Reopen and verify.",
        );
        return undefined;
      }
      setBusy("creation-discard");
      setError(null);
      const version = ++readVersion.current;
      const promise = (async () => {
        try {
          currentPending();
          const receipt: ModelCreationOperation =
            await getModelCreationOperation(operationId);
          if (
            !alive.current ||
            owner !== session.current ||
            version !== readVersion.current
          )
            return undefined;
          currentPending();
          if (receipt.operation_id !== operationId)
            throw Error("creation_receipt_changed");
          if (receipt.state === "pending") {
            const abandoned = await abandonModelCreationOperation(operationId);
            if (abandoned.state !== "deleted")
              throw Error("creation_receipt_changed");
          } else if (
            receipt.state !== "not_found" ||
            receipt.profile_id !== null ||
            receipt.created_revision !== null
          ) {
            throw Error("creation_receipt_changed");
          }
          const currentProfiles = await getModelProfiles();
          if (
            !alive.current ||
            owner !== session.current ||
            version !== readVersion.current
          )
            return undefined;
          currentPending();
          setProfiles(currentProfiles);
          setRefreshError(null);
          if (currentProfiles.length > 0) {
            setCreationMissing(null);
            setError(
              "Model configurations were found; the old creation record is kept. Please verify the existing configurations first.",
            );
            return undefined;
          }
          clearPendingModelCreation(operationId);
          if (readPendingModelCreation() !== null)
            throw Error("creation_clear_unconfirmed");
          pendingCreationRef.current = null;
          setPendingCreation(null);
          setCreationStorageFailed(false);
          setCreationMissing(null);
          setError(null);
          return true;
        } catch {
          if (alive.current && owner === session.current) {
            setCreationMissing(null);
            setError(
              "The old creation record has not been safely released; the configuration remains locked. Please verify again.",
            );
          }
          return undefined;
        } finally {
          if (alive.current && owner === session.current) {
            flight.current = null;
            setBusy(null);
          }
        }
      })();
      flight.current = { key: "creation-discard", promise };
      return promise;
    },
    refreshProfiles: () => (flight.current ? Promise.resolve() : refresh()),
    create: (input: ModelProfileInput) =>
      run(
        "create",
        async (traceId) => {
          const owner = session.current;
          let existing: string | null;
          try {
            existing = readPendingModelCreation();
          } catch {
            setCreationStorageFailed(true);
            throw Error("creation_storage_unavailable");
          }
          if (existing) {
            pendingCreationRef.current = existing;
            setPendingCreation(existing);
            throw Error("pending_creation_exists");
          }
          const operationId = crypto.randomUUID();
          try {
            savePendingModelCreation(operationId);
          } catch {
            setCreationStorageFailed(true);
            throw Error("creation_storage_unavailable");
          }
          pendingCreationRef.current = operationId;
          setPendingCreation(operationId);
          let profile: ModelProfile;
          try {
            profile = await createModelProfile(
              {
                ...input,
                creation_operation_id: operationId,
              },
              traceId,
            );
          } catch (reason) {
            // This exact Core validation response precedes handler entry and
            // reservation. Other errors and lookup failures remain uncertain.
            if (
              alive.current &&
              owner === session.current &&
              axios.isAxiosError(reason) &&
              reason.response?.status === 422 &&
              reason.response.data?.detail === "request_validation_failed"
            ) {
              clearPendingModelCreation(operationId);
              return null;
            }
            throw reason;
          }
          const receipt = await getModelCreationOperation(operationId);
          if (!alive.current || owner !== session.current)
            throw Error("creation_owner_changed");
          if (
            receipt.operation_id !== operationId ||
            receipt.state !== "committed" ||
            receipt.profile_id !== profile.profile_id ||
            receipt.created_revision !== 1
          )
            throw Error("creation_receipt_mismatch");
          clearPendingModelCreation(operationId);
          return profile;
        },
        (profile) => {
          pendingCreationRef.current = null;
          setPendingCreation(null);
          if (profile) mergeProfile(profile);
          else
            setError(
              "Configuration not saved; check the model address and parameters, then retry. Your entries are kept.",
            );
        },
        () =>
          setError(
            "The creation result is unconfirmed. Check what was saved; do not submit again.",
          ),
      ),
    probeProfile: testProfile,
    activate: (profileId: string) =>
      run(
        `activate:${profileId}`,
        (traceId) => activateModelProfile(profileId, traceId),
        mergeProfile,
      ),
    remove: (profileId: string) =>
      run(
        `delete:${profileId}`,
        (traceId) => deleteModelProfile(profileId, traceId),
        () => {
          setProfiles((previous) =>
            previous.filter((item) => item.profile_id !== profileId),
          );
          setProbeResults((previous) =>
            Object.fromEntries(
              Object.entries(previous).filter(
                ([, value]) => value.profile_id !== profileId,
              ),
            ),
          );
          setProbeErrors((previous) =>
            Object.fromEntries(
              Object.entries(previous).filter(
                ([key]) => JSON.parse(key)[0] !== profileId,
              ),
            ),
          );
        },
      ),
    update: (profileId: string, patch: ModelProfilePatch) =>
      run(
        `update:${profileId}`,
        (traceId) => updateModelProfile(profileId, patch, traceId),
        mergeProfile,
      ),
  };
}

function message(reason: unknown): string {
  if (reason instanceof Error && reason.message.startsWith("model_probe_")) {
    if (reason.message === "model_probe_operation_stale")
      return "The model configuration changed; the old test result will not be used for activation. Please verify the configuration again.";
    if (reason.message === "model_probe_capability_unavailable")
      return "The test service cannot be reached; no test was sent this time. Restore the connection and retry.";
    if (reason.message === "model_probe_other_profile_pending")
      return "Another model's test result is unconfirmed; verify that model's test result first.";
    return "The test result is unconfirmed. Clicking Test connection again verifies the original operation without sending a duplicate test.";
  }
  const code = apiErrorCode(reason);
  if (code === "model_profile_probe_required")
    return "Click Test connection first; set it as the current model once it passes.";
  if (code === "active_model_profile")
    return "This is the model currently in use. Switch to another model before deleting it.";
  if (
    code &&
    [
      "model_provider_not_supported",
      "embedding_model_required",
      "model_profile_not_found",
      "model_profile_conflict",
    ].includes(code)
  )
    return modelErrorText(code);
  // Raw transport errors can contain a request body or pasted credential.
  return "The model configuration operation did not succeed. Check your input and retry.";
}
