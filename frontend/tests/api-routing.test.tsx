import { describe, expect, it } from "vitest";

import { resolveApiBaseUrl } from "../src/api";

describe("API routing", () => {
  it("uses the Vercel same-origin proxy on production and preview hosts", () => {
    const configured = "https://desktop-ct23ruu.tail996bd3.ts.net";

    expect(resolveApiBaseUrl(configured, "phonescribe.vercel.app")).toBe("");
    expect(resolveApiBaseUrl(configured, "phonescribe-git-main-rhim.vercel.app")).toBe("");
  });

  it("keeps the configured backend for non-Vercel hosts", () => {
    expect(resolveApiBaseUrl("https://phone.example.ts.net/", "localhost")).toBe(
      "https://phone.example.ts.net"
    );
  });
});
