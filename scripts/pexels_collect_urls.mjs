import fs from 'node:fs';
export default async function run(page, ui) {
  const urls = new Set();
  for (let n = 1; n <= 12; n++) {
    await page.goto(`https://www.pexels.com/search/keke%20napep/?page=${n}`, { waitUntil: 'domcontentloaded', timeout: 30000 });
    await page.waitForTimeout(2500);
    const height = await page.evaluate(() => document.body.scrollHeight);
    for (let y = 0; y < height; y += 900) {
      await page.evaluate((v) => window.scrollTo(0, v), y);
      await page.waitForTimeout(120);
    }
    await page.waitForTimeout(1500);
    const found = await page.evaluate(() => [...document.querySelectorAll('img')]
      .map((i) => i.currentSrc || i.src)
      .filter((u) => u && u.includes('images.pexels.com/photos/'))
      .map((u) => u.split('?')[0])
      .filter((u) => !u.includes('/lib/') && !u.includes('/assets/')));
    found.forEach((u) => urls.add(u));
  }
  const out = [...urls].map((u) => u + '?auto=compress&cs=tinysrgb&w=1280');
  fs.writeFileSync('/Users/omegasesay/Downloads/freetown-traffic/data/keke_pexels_urls.json', JSON.stringify(out, null, 2));
  return { count: out.length, path: '/Users/omegasesay/Downloads/freetown-traffic/data/keke_pexels_urls.json', sample: out.slice(0, 5) };
}
