import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { Workbook } from '@oai/artifact-tool';

const original = await fs.readFile('tmp/review/review.original.csv', 'utf8');
const decisions = JSON.parse(await fs.readFile('tmp/review/decisions.json', 'utf8'));
const wb = await Workbook.fromCSV(original, {sheetName: 'Review'});
const sheet = wb.worksheets.getItem('Review');
const source = sheet.getRange('A1:I101').values;
assert.equal(source.length, 101);
for (let i = 0; i < decisions.length; i++) {
  assert.equal(source[i + 1][0], decisions[i].id);
  assert.ok(!source[i + 1][7] && !source[i + 1][8]);
}
// These widths affect only the temporary preview. CSV retains no formatting.
sheet.getRange('H1:H8').format.columnWidthPx = 150;
sheet.getRange('I1:I8').format.columnWidthPx = 830;
sheet.getRange('H1:I8').format.rowHeightPx = 60;
sheet.getRange('H1:I8').format.wrapText = true;
const before = await wb.render({sheetName: 'Review', range: 'H1:I8', scale: 1});
await fs.writeFile('tmp/review/before.png', new Uint8Array(await before.arrayBuffer()));
sheet.getRange('H2:I101').values = decisions.map(d => [d.verdict, d.note]);
wb.recalculate();
const result = sheet.getRange('A1:I101').values;
for (let i = 0; i < result.length; i++) {
  assert.deepEqual(result[i].slice(0, 7), source[i].slice(0, 7));
}
for (let i = 0; i < decisions.length; i++) {
  assert.equal(result[i + 1][7], decisions[i].verdict);
  assert.equal(result[i + 1][8], decisions[i].note);
}
console.log((await wb.inspect({kind: 'table', range: 'Review!H21:I30', include: 'values', tableMaxRows: 10, tableMaxCols: 2, tableMaxCellChars: 120, maxChars: 2500})).ndjson);
const after = await wb.render({sheetName: 'Review', range: 'H1:I8', scale: 1});
await fs.writeFile('tmp/review/after.png', new Uint8Array(await after.arrayBuffer()));
const escape = value => {
  const s = value == null ? '' : String(value);
  return /[",\r\n]/.test(s) ? '"' + s.replaceAll('"', '""') + '"' : s;
};
await fs.writeFile('finetune/data/review.csv', result.map(row => row.map(escape).join(',')).join('\r\n') + '\r\n');
console.log('Saved 100 reviewed rows. Only verdict and note values changed.');
