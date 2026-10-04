/**
 * 界面冒烟测试（无需浏览器）—— 确认 `web/index.html` 不是白屏。
 *
 * 本机没有 agent-browser（Chromium 要 500MB+），而"打开浏览器看一眼"在无人值守时不成立。
 * 这里用 Node 把页面的 <script> 抽出来、真连本地服务跑一遍，断言：
 *   结果卡片数 / 小图 svg 是否生成 / 叠图 path 条数 / 产出里有没有 NaN·undefined·Infinity
 *
 * ⚠️ 只验证**逻辑跑得通**，不验证视觉效果（颜色、间距、字体一律没测到）。
 *
 * 用法（先确保服务已在跑）：
 *   node scripts/smoke_ui.js
 *   SMOKE_BASE=http://127.0.0.1:9000 node scripts/smoke_ui.js
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const BASE = process.env.SMOKE_BASE || 'http://127.0.0.1:8765';

const html = fs.readFileSync(path.join(ROOT, 'web', 'index.html'), 'utf8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) { console.log('❌ 页面里没有 <script> 段'); process.exit(1); }

// ⚠️ new Function 的作用域是隔离的 → 要测的函数必须显式挂到 globalThis
const src = m[1] +
  '\n;globalThis.__run = run; globalThis.__render = render; globalThis.__live = runLive;';

/* ---------------- 桩对象 ---------------- */
const els = {};
const mk = sel => ({
  _sel: sel, innerHTML: '', textContent: '', value: '', checked: false, placeholder: '',
  dataset: {}, style: {}, onclick: null, onchange: null,
  classList: { add() {}, remove() {} }
});
// 同一选择器必须永远返回**同一个**桩对象，否则 checked=true 设了个寂寞
const sel = s => (els[s] = els[s] || mk(s));

global.document = Object.defineProperty({
  querySelector: sel,
  querySelectorAll: () => [],
  addEventListener: () => {}
}, 'title', { get() { return ''; }, set() {}, configurable: true });
global.window = { scrollTo() {} };

const alerts = [];
global.alert = msg => { alerts.push(String(msg)); console.log('!! ALERT:', msg); };

// ⚠️ 必须先存原始 fetch 再覆盖，否则 __realFetch 会指向自己 → 无限递归
const realFetch = globalThis.fetch;
global.fetch = (u, o) => realFetch(BASE + u, o);

/* 预置表单值（桩不解析 HTML，select 的 value 不会自己出现） */
sel('#sym').value = 'BTCUSDT';
sel('#topk').value = '24';
sel('#variant').value = 'shape';
sel('#time').value = '';

new Function(src)();

/* ---------------- 断言工具 ---------------- */
const wait = ms => new Promise(r => setTimeout(r, ms));
let fails = 0;

function check(label, ok, detail) {
  console.log(`  ${ok ? '✅' : '❌'} ${label}${detail ? '  ' + detail : ''}`);
  if (!ok) fails++;
}

function report(title, hostSel) {
  const host = sel(hostSel).innerHTML;
  const cands = (host.match(/class="cand"/g) || []).length;
  const qsvg = sel('#qchart').innerHTML;
  const c0 = sel('#c0').innerHTML;
  const ov = sel('#ovchart').innerHTML;
  const ovPaths = (ov.match(/<path/g) || []).length;
  const junk = /NaN|undefined|Infinity/.test(host + qsvg + c0 + ov);
  const liveTag = /⚡/.test(host);

  console.log(`\n--- ${title} ---`);
  check('结果卡片已生成', cands > 0, `${cands} 张`);
  check('主图 svg 已生成', /<svg/.test(qsvg));
  check('小图 svg 已生成', /<svg/.test(c0));
  check('产出无 NaN/undefined/Infinity', !junk);
  return { cands, ovPaths, liveTag, host };
}

(async () => {
  await wait(4500);                    // 等页面末尾的 async 初始化跑完

  const a = report('① 首屏（初始化自动查库内最近一段）', '#host');
  check('首屏不该带实时标记', !a.liveTag);

  console.log('\n--- ② ⚡ 实时拉取 ---');
  const t0 = Date.now();
  try { await globalThis.__live(); } catch (e) { console.log('  live 抛错:', e.message); }
  console.log(`  （耗时 ${((Date.now() - t0) / 1000).toFixed(1)}s）`);
  const b = report('② ⚡ 实时', '#host');
  check('实时结果带 ⚡ 标记', b.liveTag);

  /* 打开所有可选分支再跑一遍（只测默认路径会漏掉一半代码） */
  console.log('\n--- ③ 实时 + 打开「显示之后 24 小时」「叠图」「同期其他币」 ---');
  sel('#after').checked = true;
  sel('#ov').checked = true;
  sel('#contemp').checked = true;
  try { await globalThis.__live(); } catch (e) { console.log('  live 抛错:', e.message); }
  const c = report('③ 全开', '#host');
  check('叠图曲线已绘制', c.ovPaths > 1, `${c.ovPaths} 条 path`);

  console.log('\n--- ④ 库内查询（历史时间） ---');
  sel('#after').checked = false;
  sel('#ov').checked = false;
  sel('#time').value = '2025-03-01 12:00';
  try { await globalThis.__run({ symbol: 'BTCUSDT', time: '2025-03-01 12:00' }); }
  catch (e) { console.log('  run 抛错:', e.message); }
  const d = report('④ 库内', '#host');
  check('库内结果不该带实时标记', !d.liveTag);

  console.log('\n' + '='.repeat(58));
  if (alerts.length) {
    console.log(`⚠️ 页面上弹了 ${alerts.length} 次 alert（上面已打印）`);
  }
  console.log(fails === 0 ? '结论：✅ 界面逻辑跑得通（无白屏、无 NaN）'
    : `结论：❌ 有 ${fails} 项没通过`);
  process.exit(fails === 0 ? 0 : 1);
})();
