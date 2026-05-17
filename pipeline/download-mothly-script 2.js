const { chromium } = require("playwright");
const path = require("path");

const URL =
  "https://passagertal.dk/Embed#vfs://global/passagertal.dk/Rejsekort/Rejsekortrejser.xview";

async function wait(page, ms) { await page.waitForTimeout(ms); }
async function snap(page, label) {
  const p = path.join(__dirname, "..", "results", "monthly", "raw", "debug", `snap_${label}.png`);
  await page.screenshot({ path: p, fullPage: false });
  console.log(`📸 ${p}`);
}

(async () => {
  const browser = await chromium.launch({ headless: false, slowMo: 100 });
  const context = await browser.newContext({
    acceptDownloads: true,
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();

  console.log("Loading...");
  await page.goto(URL, { waitUntil: "domcontentloaded" });
  await wait(page, 18000);

  // ── ATTEMPT 1: "Hent data" button in the top-right corner ────────────────────
  // This is the Excel icon visible in your screenshots — try it first.
  console.log("\n── Attempt 1: Clicking top-right 'Hent data' button...");
  try {
    const downloadPromise = page.waitForEvent("download", { timeout: 15000 });

    // Try by title text first
    const hentBtn = page.locator("text=Hent data").first();
    const hentVisible = await hentBtn.isVisible().catch(() => false);

    if (hentVisible) {
      console.log("  Found 'Hent data' — clicking...");
      await hentBtn.click();
      const download = await downloadPromise;
      const filePath = path.join(__dirname, "..", "results", "monthly", "raw", "rejsekort_hentdata.xlsx");
      await download.saveAs(filePath);
      console.log("✅ Saved via Hent data:", filePath);
      await browser.close();
      return;
    } else {
      console.log("  'Hent data' not found as text, trying image/icon near top-right...");
      // Try clicking the Excel icon area (top-right of page)
      await page.mouse.click(1270, 90);
      const download = await downloadPromise;
      const filePath = path.join(__dirname, "..", "results", "monthly", "raw", "rejsekort_hentdata.xlsx");
      await download.saveAs(filePath);
      console.log("✅ Saved via icon click:", filePath);
      await browser.close();
      return;
    }
  } catch (e) {
    console.log("  Attempt 1 failed (no download triggered):", e.message.split("\n")[0]);
  }

  // ── ATTEMPT 2: Switch to crosstab/table view on the top-left chart ───────────
  // The buttons per chart are grouped in sets of 3: [info, maximize, crosstab]
  // Chart 1 uses buttons [0, 1, 2], so button[2] = crosstab for the first chart.
  console.log("\n── Attempt 2: Switching top-left chart to crosstab view...");

  // First hover over the top-left chart to ensure its toolbar is active
  await page.mouse.move(500, 300);
  await wait(page, 1000);

  // Click the crosstab button (index 2 = ui-icon-crosstab for chart 1)
  await page.evaluate(() => {
    document.querySelectorAll(".ObjectToolbarButton")[2].click();
  });
  await wait(page, 5000);
  await snap(page, "after_crosstab_click");

  // Check if VirtualGrid appeared
  const gridCount = await page.locator(".VirtualGrid").count();
  const expandCount = await page.locator(".expandIcon").count();
  console.log(`  VirtualGrid: ${gridCount}, expandIcons: ${expandCount}`);

  if (gridCount === 0) {
    console.log("  ❌ Still no VirtualGrid. Dumping all visible class names for inspection...");
    const classes = await page.evaluate(() => {
      const all = new Set();
      document.querySelectorAll("[class]").forEach(el => {
        el.className.split(" ").forEach(c => { if (c) all.add(c); });
      });
      return [...all].sort();
    });
    console.log("  All classes on page:\n ", classes.join("\n  "));
    await browser.close();
    return;
  }

  // ── Drill down: click all expand icons until none remain ─────────────────────
  console.log("\n── Drilling down...");
  for (let round = 1; round <= 15; round++) {
    const icons = await page.evaluate(() =>
      [...document.querySelectorAll(".expandIcon")]
        .filter(el => {
          const r = el.getBoundingClientRect();
          return r.width > 0 && r.height > 0 && r.top > 0 && r.top < window.innerHeight;
        })
        .map(el => {
          const r = el.getBoundingClientRect();
          return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
        })
    );

    if (icons.length === 0) {
      console.log(`  Round ${round}: no icons — done`);
      break;
    }
    console.log(`  Round ${round}: clicking ${icons.length} icon(s)`);
    for (const { x, y } of icons) {
      await page.mouse.click(x, y);
      await wait(page, 600);
    }
    await wait(page, 2000);

    // Scroll down inside the grid to reveal more rows
    await page.evaluate(() => {
      const grid = document.querySelector(".VirtualGrid");
      if (grid) grid.scrollTop += 400;
    });
    await wait(page, 1000);
  }

  await snap(page, "after_expand");

  // ── Right-click the data grid and export ─────────────────────────────────────
  console.log("\n── Right-clicking VirtualGrid to export...");
  const grid = page.locator(".VirtualGrid").first();
  await grid.waitFor({ state: "visible", timeout: 10000 });
  await grid.click({ button: "right" });
  await wait(page, 2000);
  await snap(page, "context_menu");

  // Log visible menu items
  const menuItems = await page.evaluate(() =>
    [...document.querySelectorAll("button, [role='menuitem'], li")]
      .filter(el => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; })
      .map(el => el.textContent.trim()).filter(Boolean)
  );
  console.log("  Menu items:", menuItems);

  // Try all possible export button names (Danish and English)
  const labels = [
    "Export to Excel", "Eksportér til Excel", "Eksporter til Excel",
    "Export", "Eksportér", "Download", "Hent data",
  ];
  const downloadPromise = page.waitForEvent("download", { timeout: 60000 });

  let exported = false;
  for (const label of labels) {
    const btn = page.getByRole("button", { name: label });
    if (await btn.isVisible().catch(() => false)) {
      console.log(`  Clicking: "${label}"`);
      await btn.click();
      exported = true;
      break;
    }
  }

  if (!exported) {
    // Fallback: click any menu item containing "xcel" or "xport" or "ent"
    const fallback = await page.evaluate(() => {
      const items = [...document.querySelectorAll("button, [role='menuitem'], li")];
      const match = items.find(el =>
        /xcel|xport|ent data/i.test(el.textContent)
      );
      if (match) { match.click(); return match.textContent.trim(); }
      return null;
    });
    if (fallback) {
      console.log(`  Fallback click: "${fallback}"`);
      exported = true;
    }
  }

  if (!exported) {
    console.log("❌ Could not find export button. See snap_context_menu.png");
    await browser.close();
    return;
  }

  const download = await downloadPromise;
  const filePath = path.join(__dirname, "..", "results", "monthly", "raw", "rejsekort_monthly_export_extension_data.xlsx");
  await download.saveAs(filePath);
  console.log("\n✅ Saved:", filePath);

  await browser.close();
})();