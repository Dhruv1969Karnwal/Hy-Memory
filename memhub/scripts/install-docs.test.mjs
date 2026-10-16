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

  assert.match(readme, /公共 Python SDK 已发布为 `hy-memory==2\.0\.1`/);
  assert.match(readme, /npm 包尚未发布，GitHub Releases 仍为空/s);
  assert.match(readme, /Linux x64/);
  assert.match(readme, /Linux arm64.*没有本版本公开支持证据/s);
  assert.match(readme, /macOS arm64.*正式公证.*尚未完成/s);
  assert.match(readme, /memhub uninstall/);
  assert.match(readme, /memhub delete-data/);
  assert.match(readme, /docs\/INSTALLATION\.md/);

  assert.match(installation, /384,082 bytes/);
  assert.match(installation, /277,961,300 bytes/);
  assert.match(installation, /688,414,016 bytes/);
  assert.match(installation, /277,961,849 bytes/);
  assert.match(installation, /688,414,163 bytes/);
  assert.match(installation, /fake Release/);
  assert.match(installation, /verify-fake-release-bootstrap\.mjs/);
  assert.match(installation, /N-1 升级/);
  assert.match(installation, /健康检查失败会恢复/);
  assert.match(installation, /verify-linux-upgrade-rollback\.mjs/);
  assert.match(installation, /--source/);
  assert.match(installation, /完全离线/);
  assert.match(installation, /HTTPS_PROXY/);
  assert.match(installation, /NO_PROXY/);
  assert.match(installation, /npm `postinstall` 不下载 Runtime/);
  assert.match(installation, /Python Core.*系统 Python/s);
  assert.match(installation, /默认数据目录是 `~\/\.one-memory`/);
  assert.match(installation, /Windows.*不支持/s);
  assert.match(installation, /最低 macOS 13/);
  assert.match(installation, /手动下载新版本/);
  assert.match(installation, /macos-15-intel/);
  assert.match(installation, /auto-updater/);
  assert.doesNotMatch(installation, /npm install -g @tencent\/memhub_internal/);
});
