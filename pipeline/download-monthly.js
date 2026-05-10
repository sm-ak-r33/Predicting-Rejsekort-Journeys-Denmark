const { chromium } = require("playwright");
const path = require("path");

const URL = "https://passagertal.dk/Embed#vfs://global/passagertal.dk/Rejsekort/Rejsekortrejser.xview";

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    acceptDownloads: true,
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();

  await page.goto(URL, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(15000);
  await page.mouse.move(800, 350);
  await page.waitForTimeout(1000);

  await page.evaluate(() => {
    document.querySelectorAll(".ObjectToolbarButton")[3].click();
  });
  await page.waitForTimeout(5000);

  await page.mouse.click(800, 350, { button: "right" });
  await page.waitForTimeout(1000);

  const downloadPromise = page.waitForEvent("download", { timeout: 120000 });
  await page.getByRole("button", { name: "Export to Excel" }).click();
  const download = await downloadPromise;
  await download.saveAs(path.join(__dirname, "rejsekort_monthly_chart_export.xlsx"));

  await browser.close();
})().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
