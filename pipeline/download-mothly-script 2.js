const { chromium } = require("playwright");
const path = require("path");

const URL = "https://passagertal.dk/Embed#vfs://global/passagertal.dk/Rejsekort/Rejsekortrejser.xview";

async function wait(page, ms) {
  await page.waitForTimeout(ms);
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    acceptDownloads: true,
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();

  await page.goto(URL, { waitUntil: "domcontentloaded" });
  await wait(page, 18000);

  try {
    const downloadPromise = page.waitForEvent("download", { timeout: 15000 });
    const hentBtn = page.locator("text=Hent data").first();
    const hentVisible = await hentBtn.isVisible().catch(() => false);
    if (hentVisible) {
      await hentBtn.click();
    } else {
      await page.mouse.click(1270, 90);
    }
    const download = await downloadPromise;
    await download.saveAs(path.join(__dirname, "rejsekort_monthly_export_extension_data.xlsx"));
    await browser.close();
    return;
  } catch {}

  await page.mouse.move(500, 300);
  await wait(page, 1000);

  await page.evaluate(() => {
    document.querySelectorAll(".ObjectToolbarButton")[2].click();
  });
  await wait(page, 5000);

  const gridCount = await page.locator(".VirtualGrid").count();
  if (gridCount === 0) {
    await browser.close();
    throw new Error("VirtualGrid not found");
  }

  for (let round = 1; round <= 15; round++) {
    const icons = await page.evaluate(() =>
      [...document.querySelectorAll(".expandIcon")]
        .filter((el) => {
          const r = el.getBoundingClientRect();
          return r.width > 0 && r.height > 0 && r.top > 0 && r.top < window.innerHeight;
        })
        .map((el) => {
          const r = el.getBoundingClientRect();
          return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
        })
    );
    if (icons.length === 0) break;
    for (const { x, y } of icons) {
      await page.mouse.click(x, y);
      await wait(page, 600);
    }
    await wait(page, 2000);
    await page.evaluate(() => {
      const grid = document.querySelector(".VirtualGrid");
      if (grid) grid.scrollTop += 400;
    });
    await wait(page, 1000);
  }

  const grid = page.locator(".VirtualGrid").first();
  await grid.waitFor({ state: "visible", timeout: 10000 });
  await grid.click({ button: "right" });
  await wait(page, 2000);

  const labels = [
    "Export to Excel",
    "Eksportér til Excel",
    "Eksporter til Excel",
    "Export",
    "Eksportér",
    "Download",
    "Hent data",
  ];

  const downloadPromise = page.waitForEvent("download", { timeout: 60000 });
  let exported = false;
  for (const label of labels) {
    const btn = page.getByRole("button", { name: label });
    if (await btn.isVisible().catch(() => false)) {
      await btn.click();
      exported = true;
      break;
    }
  }

  if (!exported) {
    const fallback = await page.evaluate(() => {
      const items = [...document.querySelectorAll("button, [role='menuitem'], li")];
      const match = items.find((el) => /xcel|xport|ent data/i.test(el.textContent));
      if (match) {
        match.click();
        return true;
      }
      return false;
    });
    exported = fallback;
  }

  if (!exported) {
    await browser.close();
    throw new Error("Export button not found");
  }

  const download = await downloadPromise;
  await download.saveAs(path.join(__dirname, "rejsekort_monthly_export_extension_data.xlsx"));
  await browser.close();
})().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
