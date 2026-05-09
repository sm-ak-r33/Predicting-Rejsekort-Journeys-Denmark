const { chromium } = require("playwright");
const path = require("path");

const URL =
  "https://passagertal.dk/Embed#vfs://global/passagertal.dk/Rejsekort/Rejsekortrejser.xview";

async function getFrame(page) {
  for (const frame of page.frames()) {
    try {
      const count = await frame.evaluate(() =>
        document.querySelectorAll(".ObjectToolbarButton, .VirtualGrid, .expandIcon").length
      );
      if (count > 0) return frame;
    } catch {}
  }

  return page.mainFrame();
}

async function getVisiblePlusIcons(frame) {
  return await frame.evaluate(() => {
    return [...document.querySelectorAll(".expandIcon")]
      .filter(el => {
        const r = el.getBoundingClientRect();

        if (
          r.width === 0 ||
          r.height === 0 ||
          r.top < 0 ||
          r.bottom > window.innerHeight ||
          r.left < 0 ||
          r.right > window.innerWidth
        ) {
          return false;
        }

        const rects = [...el.querySelectorAll("rect")];

        const hasVerticalBar = rects.some(rect => {
          const h = parseFloat(rect.getAttribute("height") || "0");
          const w = parseFloat(rect.getAttribute("width") || "0");
          return h > w;
        });

        return hasVerticalBar;
      })
      .map(el => {
        const r = el.getBoundingClientRect();
        return {
          x: r.left + r.width / 2,
          y: r.top + r.height / 2,
          left: Math.round(r.left),
        };
      });
  });
}

async function clickTopVisibleYearPlus(frame, page) {
  const icons = await getVisiblePlusIcons(frame);

  const yearIcons = icons
    .filter(icon => icon.left < 265)
    .sort((a, b) => a.y - b.y);

  if (yearIcons.length === 0) {
    throw new Error("No visible year-level plus icon found");
  }

  const icon = yearIcons[0];

  await page.mouse.click(icon.x, icon.y);
  await page.waitForTimeout(2500);
}

async function clickVisibleMonthPlusesBottomUp(frame, page) {
  const icons = await getVisiblePlusIcons(frame);

  const monthIcons = icons
    .filter(icon => icon.left >= 265)
    .sort((a, b) => b.y - a.y);

  for (const icon of monthIcons) {
    await page.mouse.click(icon.x, icon.y);
    await page.waitForTimeout(1000);
  }
}

async function exportExcel(page, frame, outputName) {
  const gridPos = await frame.evaluate(() => {
    const g = document.querySelector(".VirtualGrid");
    if (!g) return null;

    const r = g.getBoundingClientRect();

    return {
      x: r.left + r.width / 2,
      y: r.top + Math.min(200, r.height / 2),
    };
  });

  if (!gridPos) {
    throw new Error("VirtualGrid not found");
  }

  await page.mouse.click(gridPos.x, gridPos.y, { button: "right" });
  await page.waitForTimeout(1500);

  const downloadPromise = page.waitForEvent("download", {
    timeout: 120000,
  });

  const clickedExport = await frame.evaluate(() => {
    const item = [...document.querySelectorAll("button, div, span, li")]
      .filter(el => {
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0;
      })
      .find(el => el.textContent.trim() === "Export to Excel");

    if (!item) return false;

    item.click();
    return true;
  });

  if (!clickedExport) {
    throw new Error("Export to Excel not found");
  }

  const download = await downloadPromise;
  const out = path.join(__dirname, outputName);
  await download.saveAs(out);

  return out;
}

(async () => {
  const browser = await chromium.launch({
    headless: true,
  });

  const context = await browser.newContext({
    acceptDownloads: true,
    viewport: { width: 1600, height: 1000 },
  });

  const page = await context.newPage();

  console.log("Loading dashboard...");
  await page.goto(URL, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(18000);

  let frame = await getFrame(page);

  console.log("Switching to crosstab...");
  await frame.evaluate(() => {
    const btn =
      document.querySelector("div:nth-child(4) > .ObjectToolbarButton") ||
      [...document.querySelectorAll(".ObjectToolbarButton")]
        .find(b => b.className.includes("crosstab"));

    if (!btn) {
      throw new Error("Crosstab button not found");
    }

    btn.click();
  });

  await page.waitForTimeout(6000);
  frame = await getFrame(page);

  console.log("Sorting latest year to top...");
  await page.mouse.click(415, 218);
  await page.waitForTimeout(1500);

  await page.mouse.click(425, 212);
  await page.waitForTimeout(3000);

  console.log("Expanding latest year...");
  await clickTopVisibleYearPlus(frame, page);

  console.log("Expanding months...");
  await clickVisibleMonthPlusesBottomUp(frame, page);

  console.log("Exporting Excel...");
  const savedPath = await exportExcel(
    page,
    frame,
    "rejsekort_latest_year_daily_export.xlsx"
  );

  console.log("Saved:", savedPath);

  await browser.close();
})();