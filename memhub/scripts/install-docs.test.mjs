import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("installation docs disclose publication and lifecycle boundaries", async () => {
  const readme = await readFile(
    new URL("../README.md", import.meta.url),
    "utf8",
  );
  const installation = await readFile(
    new URL("../docs/INSTALLATION.md", import.meta.url),
    "utf8",
  );

  assert.match(readme, /The public Python SDK is `hy-memory==2\.0\.3`/);
  assert.match(
    readme,
    /The release targets of `v1\.0\.3` are the Linux x64\/arm64 Runtime/,
  );
  assert.match(readme, /Linux artifacts are built on GitHub Actions/);
  assert.match(readme, /macOS.*Developer ID signing, notarization/s);
  assert.match(readme, /release-manifest\.json/);
  assert.match(readme, /memhub uninstall/);
  assert.match(readme, /memhub delete-data/);
  assert.match(readme, /docs\/INSTALLATION\.md/);
  assert.doesNotMatch(readme, /private RC|internal candidate/);

  assert.match(installation, /Linux x64.*GitHub Actions/s);
  assert.match(installation, /Linux arm64.*GitHub Actions/s);
  assert.match(installation, /macOS arm64.*Developer ID/s);
  assert.match(installation, /macOS x64.*Developer ID/s);
  assert.match(installation, /release-manifest\.json/);
  assert.match(installation, /fake Release/);
  assert.match(installation, /--source/);
  assert.match(installation, /fully offline/);
  assert.match(installation, /HTTPS_PROXY/);
  assert.match(installation, /NO_PROXY/);
  assert.match(installation, /npm `postinstall` does not download the Runtime/);
  assert.match(installation, /Python Core.*system Python/s);
  assert.match(installation, /default data directory is `~\/\.one-memory`/);
  assert.match(
    installation,
    /Windows is not supported|Currently not supported/s,
  );
  assert.match(installation, /minimum supported macOS version is 13/);
  assert.match(
    installation,
    /manually download the new version from GitHub Releases/,
  );
  assert.match(installation, /auto-updater/);
  assert.doesNotMatch(installation, /private RC|internal candidate/);
  assert.doesNotMatch(installation, /npm install -g @tencent\/memhub_internal/);
});
