import test from "node:test";
import assert from "node:assert/strict";

import { isObject, normalizeBranchId, rk } from "../server.js";

test("rk namespaces sockets by branch and role", () => {
  assert.equal(rk("seocho-01", "agent"), "seocho-01:agent");
  assert.equal(rk(null, "kiosk"), "_default:kiosk");
});

test("normalizeBranchId accepts only compact branch identifiers", () => {
  assert.equal(normalizeBranchId("gangnam_01"), "gangnam_01");
  assert.equal(normalizeBranchId(""), "_default");
  assert.equal(normalizeBranchId("../agent"), null);
  assert.equal(normalizeBranchId("branch with spaces"), null);
});

test("isObject rejects arrays and nulls", () => {
  assert.equal(isObject({ role: "agent" }), true);
  assert.equal(isObject([]), false);
  assert.equal(isObject(null), false);
});
