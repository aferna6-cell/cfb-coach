// Drives the REAL established live page (served by the unified HTML server) in headless Chromium.
// usage: node browser_drive.js <url> <outdir>
const { chromium } = require('playwright');
const fs = require('fs');

(async () => {
  const [url, outdir] = process.argv.slice(2);
  fs.mkdirSync(outdir, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1100, height: 1500 } });
  const out = { steps: [], errors: [], console: [] };
  page.on('pageerror', e => out.errors.push(String(e)));
  page.on('console', m => { if (m.type() === 'error') out.console.push(m.text()); });
  const T = async (sel) => (await page.locator(sel).first().innerText()).trim();
  const submit = async () => {
    const t0 = Date.now();
    const [resp] = await Promise.all([
      page.waitForResponse(r => r.url().endsWith('/api/result_call') && r.request().method() === 'POST'),
      page.click('#btn-submit'),
    ]);
    const ok = (await resp.json()).ok;
    await page.waitForFunction(() => !document.getElementById('btn-submit').disabled);
    return { ok, ms: Date.now() - t0 };
  };
  const snapshot = async (label, extra = {}) => {
    const s = {
      label, call: await T('#call'), panel: (await page.locator('#unified-panel').textContent()).trim(), resultNote: await T('#unified-result-note'),
      qbReads: (await page.locator('#qb-reads-list li').allInnerTexts()), qbVisible: await page.locator('#qb-reads').isVisible(),
      user: (await page.locator('#unified-panel .u-user').count()) ? await T('#unified-panel .u-user') : null,
      log: await page.locator('#log div').allTextContents(), err: await T('#err'), ...extra,
    };
    out.steps.push(s);
    return s;
  };
  const setSit = async (side, down, dist, ylSide, yl) => {
    await page.selectOption('#side', side);
    await page.fill('#down', String(down)); await page.fill('#distance', String(dist));
    await page.selectOption('#yl-side', ylSide); await page.fill('#yl', String(yl));
  };

  await page.goto(url);
  await page.waitForFunction(() => document.getElementById('unified-panel') && !document.getElementById('unified-panel').hidden);
  await snapshot('initial');

  // 1. offense snap
  await setSit('offense', 1, 10, 'my', 25);
  let r = await submit();
  await snapshot('offense-1', r);

  // 2. result of snap 1: standing attestation confirms recommended by default
  await page.click('button.outcome[data-out="gain"]'); await page.fill('#gain-n', '7');
  await page.fill('#their', '4-3 Over | Cover 3 Sky');
  await setSit('offense', 2, 3, 'my', 32);
  r = await submit();
  await snapshot('offense-2', r);

  // 3. possession change -> defense. Explicitly override standing attestation to UNKNOWN for snap 2.
  await page.click('button.outcome[data-out="gain"]'); await page.fill('#gain-n', '5');
  await page.click('#btn-ran-different');
  await page.check('input[name="exec"][value="unknown"]');
  await setSit('defense', 1, 10, 'my', 40);
  r = await submit();
  await snapshot('defense-3', r);

  // 4. result of defensive snap = OPPONENT offense's yards; implicit recommended; back to offense
  await page.click('button.outcome[data-out="gain"]'); await page.fill('#gain-n', '12');
  await page.fill('#their', 'Gun Trips X Nasty | Hi Lo Cross');
  await setSit('offense', 1, 10, 'my', 30);
  r = await submit();
  await snapshot('offense-4', r);

  // 5. refresh / reconnect: the waiting snap must still be there, no new snap created
  const before = (await page.locator('#unified-panel').textContent()).trim();
  await page.reload();
  await page.waitForFunction(() => document.getElementById('call').textContent.indexOf('—') > 0);
  await snapshot('after-reload', { panelBefore: before });
  await page.screenshot({ path: `${outdir}/unified-after-reload.png`, fullPage: true });

  // 6. bad input must show an error and change nothing
  await page.fill('#down', '9');
  const resp = await Promise.all([page.waitForResponse(x => x.url().endsWith('/api/result_call')), page.click('#btn-submit')]);
  await page.waitForTimeout(150);
  await snapshot('bad-input', { ok: (await resp[0].json()).ok });

  await browser.close();
  process.stdout.write(JSON.stringify(out));
})().catch(e => { console.error('BROWSER-DRIVE-FAILED', e); process.exit(1); });
