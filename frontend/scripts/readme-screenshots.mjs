// Regenerates the README screenshots from readme-scenario.json against a running app.
// Usage: npm run screenshots   (the app must be serving at SCREENSHOT_URL, default the Vite dev server)

import { readFileSync } from "node:fs";
import { mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { deflateRawSync } from "node:zlib";

import { chromium } from "playwright-core";

const here = dirname(fileURLToPath(import.meta.url));
const OUT = join(here, "..", "..", "media");
const BASE = process.env.SCREENSHOT_URL ?? "http://localhost:5173";
const WIDTH = 1240;
const MARGIN = 20;

const scenario = JSON.parse(readFileSync(join(here, "readme-scenario.json"), "utf8"));

async function link() {
  const form = await (await fetch(`${BASE}/api/explore/form`)).json();
  const snapshot = { registry: form.registry_version, ...scenario };
  const packed = deflateRawSync(Buffer.from(JSON.stringify(snapshot)));
  return `v1.${packed.toString("base64url")}`;
}

async function frame(page, selectors, file) {
  await page.evaluate((all) => {
    const kept = all.map((s) => document.querySelector(s));
    for (const block of document.querySelectorAll(".wrap > *, .outcome > *")) {
      const inFrame = kept.some((k) => block.contains(k) || k.contains(block));
      block.style.visibility = inFrame ? "" : "hidden";
    }
  }, selectors);
  const box = await page.evaluate((all) => {
    const rects = all.map((s) => document.querySelector(s).getBoundingClientRect());
    const top = Math.min(...rects.map((r) => r.top));
    const bottom = Math.max(...rects.map((r) => r.bottom));
    const left = Math.min(...rects.map((r) => r.left));
    const right = Math.max(...rects.map((r) => r.right));
    return { x: left, y: top + scrollY, width: right - left, height: bottom - top };
  }, selectors);
  await page.screenshot({
    path: join(OUT, file),
    fullPage: true,
    clip: {
      x: box.x - MARGIN,
      y: box.y - MARGIN,
      width: box.width + 2 * MARGIN,
      height: box.height + 2 * MARGIN,
    },
  });
  console.log(`wrote media/${file}`);
}

async function shoot(browser, code, theme) {
  const page = await browser.newPage({
    viewport: { width: WIDTH, height: 900 },
    deviceScaleFactor: 2,
    colorScheme: theme,
    reducedMotion: "reduce",
  });
  await page.goto(`${BASE}/#${code}`);
  await page.waitForSelector(".result .rank");
  await page.waitForSelector("table.matrix");
  if (await page.locator(".msg.error").count()) throw new Error("the page reported an error");

  await page.evaluate(() => {
    const [setup, alternatives] = document.querySelectorAll(".wrap > .card");
    setup.id = "shot-setup";
    alternatives.id = "shot-alternatives";
    const [result, ranking] = document.querySelectorAll(".outcome > section");
    result.id = "shot-result";
    ranking.id = "shot-ranking";
  });

  await page.evaluate(() => {
    const families = [...document.querySelectorAll("#shot-alternatives tbody tr.familyrow")];
    const cut = families[2];
    if (!cut) return;
    for (let row = cut; row; row = row.nextElementSibling) row.style.display = "none";
  });
  await frame(page, ["#shot-setup", "#shot-alternatives"], `inputs-${theme}.png`);
  await page.evaluate(() => {
    for (const row of document.querySelectorAll("#shot-alternatives tbody tr")) row.style.display = "";
  });
  await frame(page, ["#shot-result", "#shot-ranking"], `ranking-by-application-${theme}.png`);
  await page.getByRole("tab", { name: "By priorities" }).click();
  await page.waitForFunction(() => document.querySelectorAll("table.matrix tbody tr").length > 6);
  await frame(page, ["#shot-ranking"], `ranking-by-priorities-${theme}.png`);
  await page.close();
}

await mkdir(OUT, { recursive: true });
const code = await link();
const browser = await chromium.launch({ channel: "chrome" });
try {
  for (const theme of ["light", "dark"]) await shoot(browser, code, theme);
} finally {
  await browser.close();
}
