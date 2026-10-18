// Log export is offered from two places (the health view and the settings
// help card). Both must report the same outcome wording, so the desktop
// bridge call and the message live here instead of being copied.
export type DiagnosticsExportResult = {
  ok: boolean;
  message: string;
};

export async function exportDiagnosticsFile(): Promise<DiagnosticsExportResult> {
  if (!window.oneMemory?.exportDiagnostics) {
    return { ok: false, message: "Export diagnostics from the desktop app." };
  }
  try {
    const receipt = await window.oneMemory.exportDiagnostics();
    return {
      ok: true,
      message: `Log exported: ${receipt.filename ?? "Diagnostics file"}`,
    };
  } catch {
    return { ok: false, message: "Failed to export diagnostics. Try again." };
  }
}
