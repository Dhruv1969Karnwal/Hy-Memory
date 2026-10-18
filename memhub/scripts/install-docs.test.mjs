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

  assert.match(readme, /\u516c\u5171 Python SDK \u4e3a `hy-memory==2\.0\.3`/);
  assert.match(
    readme,
    /`v1\.0\.3` \u7684\u53d1\u884c\u76ee\u6807\u662f Linux x64\/arm64 Runtime/,
  );
  assert.match(readme, /Linux \u5236\u54c1\u7531 GitHub Actions/);
  assert.match(readme, /macOS.*Developer ID \u7b7e\u540d\u3001\u516c\u8bc1/s);
  assert.match(readme, /release-manifest\.json/);
  assert.match(readme, /memhub uninstall/);
  assert.match(readme, /memhub delete-data/);
  assert.match(readme, /docs\/INSTALLATION\.md/);
  assert.doesNotMatch(readme, /\u79c1\u6709 RC|\u5185\u90e8\u5019\u9009/);

  assert.match(installation, /Linux x64.*GitHub Actions/s);
  assert.match(installation, /Linux arm64.*GitHub Actions/s);
  assert.match(installation, /macOS arm64.*Developer ID/s);
  assert.match(installation, /macOS x64.*Developer ID/s);
  assert.match(installation, /release-manifest\.json/);
  assert.match(installation, /fake Release/);
  assert.match(installation, /--source/);
  assert.match(installation, /\u5b8c\u5168\u79bb\u7ebf/);
  assert.match(installation, /HTTPS_PROXY/);
  assert.match(installation, /NO_PROXY/);
  assert.match(installation, /npm `postinstall` \u4e0d\u4e0b\u8f7d Runtime/);
  assert.match(installation, /Python Core.*\u7cfb\u7edf Python/s);
  assert.match(
    installation,
    /\u9ed8\u8ba4\u6570\u636e\u76ee\u5f55\u662f `~\/\.one-memory`/,
  );
  assert.match(installation, /Windows.*\u4e0d\u652f\u6301/s);
  assert.match(installation, /\u6700\u4f4e\u652f\u6301 macOS 13/);
  assert.match(installation, /\u624b\u52a8\u4e0b\u8f7d\u65b0\u7248\u672c/);
  assert.match(installation, /auto-updater/);
  assert.doesNotMatch(installation, /\u79c1\u6709 RC|\u5185\u90e8\u5019\u9009/);
  assert.doesNotMatch(installation, /npm install -g @tencent\/memhub_internal/);
});
