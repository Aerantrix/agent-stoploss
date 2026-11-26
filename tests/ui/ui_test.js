const { JSDOM, VirtualConsole } = require("jsdom");
const BASE = process.env.STOPLOSS_URL || "http://127.0.0.1:8741/";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = [];
function check(name, cond, extra) { results.push([cond ? "PASS" : "FAIL", name, extra || ""]); }

(async () => {
  const html = await (await fetch(BASE)).text();
  const errors = [];
  const vc = new VirtualConsole();
  vc.on("jsdomError", (e) => errors.push(String(e.message || e)));
  vc.on("error", (e) => errors.push(String(e)));
  const dom = new JSDOM(html, {
    url: BASE, runScripts: "dangerously", pretendToBeVisual: true, virtualConsole: vc,
    beforeParse(w) {
      w.fetch = (p, o) => fetch(new URL(p, BASE), o);
      w.HTMLDialogElement.prototype.showModal = function () { this.setAttribute("open", ""); };
      w.HTMLDialogElement.prototype.close = function () { this.removeAttribute("open"); };
    },
  });
  const w = dom.window, d = w.document;
  const until = async (fn, ms = 8000) => { const t = Date.now(); while (Date.now() - t < ms) { try { if (fn()) return true; } catch (e) {} await sleep(50); } return false; };
  const text = (sel) => (d.querySelector(sel) || {}).textContent || "";
  const tileVal = (label) => { for (const t of d.querySelectorAll(".tile")) if (t.querySelector(".label").textContent.includes(label)) return t.querySelector(".value").textContent; };
  const tileBase = (label) => { for (const t of d.querySelectorAll(".tile")) if (t.querySelector(".label").textContent.includes(label)) return (t.querySelector(".base") || {}).textContent; };
  const fire = (node, type) => node.dispatchEvent(new w.Event(type, { bubbles: true }));

  check("summary renders", await until(() => d.querySelectorAll(".tile").length === 6));
  check("header", text("#src").startsWith("660 runs"), text("#src"));
  check("runs tile 660", tileVal("Runs") === "660");
  check("would stop 251", tileVal("Would stop") === "251");
  check("breaks 6 good runs", tileVal("Good runs it would break") === "6");
  check("missed 8", tileVal("Missed") === "8");
  check("share 99%", tileVal("Share of waste saved") === "99%");
  check("mechanism bars 4", d.querySelectorAll(".bars .bar").length === 4);
  check("not dirty", text("#state").startsWith("Same as"));

  // runs tab
  d.querySelector('[data-tab="runs"]').click();
  await until(() => d.querySelectorAll("#runsTable tbody tr").length);
  check("runs table 200 rows", d.querySelectorAll("#runsTable tbody tr").length === 200);
  check("show more", [...d.querySelectorAll("#runsTable button")].some((b) => b.textContent.includes("460 left")));
  const chip = [...d.querySelectorAll(".chip")].find((c) => c.textContent.startsWith("Would break ok"));
  chip.click();
  await sleep(50);
  check("filter: break ok = 6", d.querySelectorAll("#runsTable tbody tr").length === 6);
  const search = d.querySelector('#tab-runs input[type=search]');
  search.value = "s06-A-r1-strict"; fire(search, "input"); await sleep(30);
  check("search narrows", d.querySelectorAll("#runsTable tbody tr").length === 1);
  // sort by tokens
  [...d.querySelectorAll("#runsTable th")].find((t) => t.textContent.startsWith("Tokens"));

  // open a run
  d.querySelector("#runsTable tbody tr").click();
  check("run detail loads", await until(() => d.querySelector("#tab-run svg")));
  check("stop label in chart", text("#tab-run svg .stoplabel").includes("Tool keeps failing"), text("#tab-run svg .stoplabel"));
  check("rows after stop greyed", d.querySelectorAll("#tab-run tr.after").length > 0);
  check("one stop verdict row", [...d.querySelectorAll("#tab-run td")].filter((t) => t.textContent.startsWith("Stop ·")).length === 1);
  check("events listed", d.querySelectorAll("#tab-run tbody tr").length > 10);

  // change thresholds: breaker 5 of 5 -> the 3 flapping runs no longer broken
  const failRange = [...d.querySelectorAll("#controls .row")].find((r) => r.textContent.includes("Failures in window")).querySelector("input");
  failRange.value = "5"; fire(failRange, "input");
  check("dirty after change", text("#state").startsWith("Changed"));
  // with 5 of 5 the flapping run is no longer stopped by the breaker; it
  // repeats a call later and the loop detector stops it at step 32
  await until(() => text("#tab-run .facts").includes("Same call repeated"), 4000);
  check("run detail recomputed: now a later loop stop", text("#tab-run .facts").includes("step 32 · Same call repeated"), text("#tab-run .facts").slice(0, 220));
  d.querySelector('[data-tab="summary"]').click();
  await sleep(100);
  check("breaker stops gone from bars", ![...d.querySelectorAll(".bars .name")].some((n) => n.textContent === "Tool keeps failing"));
  check("same stop count, so no baseline", tileVal("Would stop") === "251" && !tileBase("Would stop"));
  check("saving changed + baseline", (tileBase("Would save") || "").startsWith("config file: 2,312,156"), tileBase("Would save"));

  // per-tool exemption
  const toolRow = [...d.querySelectorAll(".tools-table tbody tr")].find((r) => r.textContent.startsWith("analytics"));
  check("tool row present", !!toolRow);
  const loopBox = toolRow.querySelectorAll("input[type=checkbox]")[0];
  loopBox.checked = false; fire(loopBox, "change");
  await until(() => tileVal("Good runs it would break") === "0", 4000);
  check("loop off for analytics: 0 broken", tileVal("Good runs it would break") === "0", tileVal("Good runs it would break"));
  check("baseline shown", (tileBase("Good runs it would break") || "").includes("config file: 6"), tileBase("Good runs it would break"));
  const thrInput = toolRow.querySelector("input[type=number]");
  thrInput.value = "6"; fire(thrInput, "input");

  // invalid value -> problems, numbers kept
  const maxTok = [...d.querySelectorAll("#controls .row")].find((r) => r.textContent.includes("Max tokens")).querySelector("input");
  maxTok.value = "0"; fire(maxTok, "input");
  check("problem shown", await until(() => text("#problems").includes("run_budget.max_tokens")), text("#problems"));
  maxTok.value = "60000"; fire(maxTok, "input");
  check("problem cleared", await until(() => text("#problems") === ""));

  // export
  d.getElementById("export").click();
  check("export dialog", await until(() => d.getElementById("exportDlg").hasAttribute("open")));
  const y = text("#yamlText");
  check("yaml has changes", y.includes("max_failures: 5") && y.includes("max_tokens: 60000") && y.includes("analytics:") && y.includes("enabled: false") && y.includes("threshold: 6"), y);

  // mode switch, reset
  [...d.querySelectorAll(".seg button")].find((b) => b.textContent === "enforce").click();
  d.getElementById("reset").click();
  await sleep(300);
  check("reset", text("#state").startsWith("Same as") && tileVal("Good runs it would break") === "6");

  check("no JS errors", errors.length === 0, errors.join(" | "));
  for (const r of results) console.log(r.join("  "));
  console.log(results.filter((r) => r[0] === "FAIL").length ? "SOME FAILED" : "ALL PASSED");
  w.close();
  process.exit(0);
})().catch((e) => { console.error("CRASH", e); process.exit(1); });
