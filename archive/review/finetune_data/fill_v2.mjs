import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { Workbook } from '@oai/artifact-tool';

const original = await fs.readFile('tmp/review/v2.original.csv', 'utf8');
const checks = JSON.parse(await fs.readFile('tmp/review/v2_checks.json', 'utf8'));
const wb = await Workbook.fromCSV(original, {sheetName: 'Review'});
const sheet = wb.worksheets.getItem('Review');
const source = sheet.getRange('A1:J26').values;
assert.equal(source.length, 26);
const details = {
  bank_dedup_0351: 'education and balance are excluded only from the key, and the first row is explicitly requested.',
  bank_dedup_0462: 'age and poutcome are excluded only from the key, and the first row is explicitly requested.',
  imdb_dedup_0286: 'The single-key wording is corrected. No_of_Votes is the key and the last row is explicitly requested.',
  imdb_dedup_0521: 'Series_Title is excluded only from the key, and the first row is explicitly requested.',
  superstore_dedup_0109: 'Order Date and Product Name form the key, and the first row is explicitly requested.',
  walmart_dedup_0148: 'Weekly_Sales and Fuel_Price form the key, and the first row is explicitly requested. Null key values follow pandas grouping semantics.',
  walmart_dedup_0358: 'Temperature and Fuel_Price are excluded only from the key, and the first row is explicitly requested.',
  walmart_dedup_0412: 'Weekly_Sales is excluded only from the key, and the first row is explicitly requested.',
  air_quality_dedup_0369: 'O3 and HealthImpactClass are excluded only from the key, and the first row is explicitly requested.',
  avocado_dedup_0369: 'region is excluded only from the key, and the first row is explicitly requested.',
  bank_dedup_0161: 'default and balance form the key, and the first row is explicitly requested. Null balance values follow pandas grouping semantics.',
  diamonds_dedup_0008: 'The key includes every column, so only exact duplicate rows are removed, keeping the first.',
  diamonds_dedup_0088: 'price, x and z form the key, and the first row is explicitly requested. Null price values follow pandas grouping semantics.',
  diamonds_dedup_0334: 'Total depth percentage correctly resolves to depth, excluded only from the key. The first row is retained.',
  imdb_dedup_0135: 'The revised single-key wording is natural. Series_Title is the key and the first row is retained.',
  imdb_dedup_0358: 'Title and number of votes resolve to Series_Title and No_of_Votes, excluded only from the key. The first row is retained.',
  imdb_dedup_0472: 'Released_Year and lead actor resolve to Released_Year and Star1, excluded only from the key. The first row is retained.',
  student_dedup_0018: 'reason, absences and G1 form the key, and the first row is explicitly requested. Null absences values follow pandas grouping semantics.',
  superstore_dedup_0213: 'Customer ID and Postal Code form the key, and the first row is explicitly requested. Null postal codes follow pandas grouping semantics.',
  superstore_dedup_0232: 'The revised single-key wording is natural. Order ID is the key and the first row is retained.',
  superstore_dedup_0366: 'Product ID is excluded only from the key, and the first row is explicitly requested.',
  superstore_dedup_0372: 'State is excluded only from the key, and the first row is explicitly requested.',
  walmart_dedup_0384: 'Date and Holiday_Flag are excluded only from the key, and the first row is explicitly requested.',
};

const decisions = checks.map((check, i) => {
  const row = source[i + 1];
  assert.equal(check.id, row[0]);
  assert.equal(check.reference, 'PASS');
  assert.equal(check.noop, 'VIOLATION');
  assert.ok(!row[8] && !row[9]);
  if (check.id === 'superstore_filt_0105') {
    assert.equal(check.drop_nulls, 'VIOLATION');
    return ['unnatural_request', `Contract and reference are correct: Sales<29 rows are removed, values of 29 are retained, and all ${check.nulls_before} missing Sales rows survive. The revision still says 'a missing sales', which is unnatural. Use: 'Discard orders with sales below 29. Keep orders with missing sales values.'`];
  }
  if (check.id === 'air_quality_filt_0111') {
    assert.equal(check.drop_nulls, 'VIOLATION');
    return ['unnatural_request', `Contract and reference are correct: Humidity<53 rows are removed, values of 53 are retained, and all ${check.nulls_before} missing Humidity rows survive. 'A missing humidity' is unnatural. Use: 'Discard readings with humidity below 53. Keep readings with missing humidity values.'`];
  }
  for (const kind of ['empty', 'one_row', 'missing_group', 'reversed_survivors']) assert.equal(check[kind], 'VIOLATION');
  assert.ok(details[check.id]);
  return ['OK', details[check.id] + ' Contract and reference agree. The keep rule now enforces one survivor per original key group. Verified rejection of empty output, one-row output, a dropped group, and reversed survivor choices.'];
});

sheet.getRange('I1:I8').format.columnWidthPx = 165;
sheet.getRange('J1:J8').format.columnWidthPx = 850;
sheet.getRange('I1:J8').format.rowHeightPx = 105;
sheet.getRange('I1:J8').format.wrapText = true;
const before = await wb.render({sheetName: 'Review', range: 'I1:J8', scale: 1});
await fs.writeFile('tmp/review/v2_before.png', new Uint8Array(await before.arrayBuffer()));
sheet.getRange('I2:J26').values = decisions;
wb.recalculate();
const result = sheet.getRange('A1:J26').values;
assert.deepEqual(result[0], source[0]);
for (let i = 0; i < decisions.length; i++) {
  assert.deepEqual(result[i + 1].slice(0, 8), source[i + 1].slice(0, 8));
  assert.deepEqual(result[i + 1].slice(8), decisions[i]);
}
const after = await wb.render({sheetName: 'Review', range: 'I1:J8', scale: 1});
await fs.writeFile('tmp/review/v2_after.png', new Uint8Array(await after.arrayBuffer()));
const escape = value => {
  const s = value == null ? '' : String(value);
  return /[",\r\n]/.test(s) ? '"' + s.replaceAll('"', '""') + '"' : s;
};
await fs.writeFile('finetune/data/v2/review_changed.csv', result.map(row => row.map(escape).join(',')).join('\r\n') + '\r\n');
console.log('Saved 25 rows: 23 OK, 2 unnatural_request, 0 wrong_contract.');
