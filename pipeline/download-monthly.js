const { chromium } = require("playwright");
const path = require("path");

const URL =
  "https://passagertal.dk/Embed#vfs://global/passagertal.dk/Rejsekort/Rejsekortrejser.xview";

(async () => {
  const browser = await chromium.launch({
    headless: false,
    slowMo: 150,
  });

  const context = await browser.newContext({
    acceptDownloads: true,
    viewport: { width: 1600, height: 1000 },
  });

  const page = await context.newPage();

  await page.goto(URL, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(15000);

  // Hover/activate chart
  await page.mouse.move(800, 350);
  await page.waitForTimeout(1000);

  // This is the button that successfully drills down to daily granularity
  await page.evaluate(() => {
    document.querySelectorAll(".ObjectToolbarButton")[3].click();
  });

  await page.waitForTimeout(5000);

  // Right-click the daily chart area, not the table
  await page.mouse.click(800, 350, { button: "right" });
  await page.waitForTimeout(1000);

  const downloadPromise = page.waitForEvent("download", { timeout: 120000 });

  await page.getByRole("button", { name: "Export to Excel" }).click();

  const download = await downloadPromise;

  const filePath = path.join(__dirname, "..", "results", "monthly", "raw", "rejsekort_monthly_chart_export.xlsx");
  await download.saveAs(filePath);

  console.log("Saved:", filePath);

  await browser.close();
})();