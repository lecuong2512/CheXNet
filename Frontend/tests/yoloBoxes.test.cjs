const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');
const source = fs.readFileSync(path.resolve(__dirname, '../src/utils/yoloBoxes.ts'), 'utf8');
const code = ts.transpileModule(source, {compilerOptions:{module:ts.ModuleKind.CommonJS}}).outputText;
const exported = {};
vm.runInNewContext(code, {exports:exported});
const {yoloBoxes} = exported;
const detection = bbox => ({disease:'Cardiomegaly',confidence:0.9,cascade_score:0.6,bbox});

test('keeps coordinates on a non-square original image', () => {
  const [result] = yoloBoxes([detection({x1:300,y1:200,x2:900,y2:600})],1200,800);
  assert.deepEqual(JSON.parse(JSON.stringify(result.bbox)),{x1:300,y1:200,x2:900,y2:600});
});
test('clips partially out-of-image boxes to the original image', () => {
  const [result] = yoloBoxes([detection({x1:-15,y1:100,x2:1400,y2:900})],1200,800);
  assert.deepEqual(JSON.parse(JSON.stringify(result.bbox)),{x1:0,y1:100,x2:1200,y2:800});
});
test('rejects non-finite, reversed, empty and fully out-of-image boxes', () => {
  for (const bbox of [{x1:NaN,y1:0,x2:50,y2:40},{x1:10,y1:0,x2:5,y2:10},
    {x1:5,y1:0,x2:5,y2:10},{x1:1400,y1:0,x2:1600,y2:100}]) {
    assert.equal(yoloBoxes([detection(bbox)],1200,800).length,0);
  }
});
test('does not draw boxes before image dimensions or valid scores are available', () => {
  const box = detection({x1:10,y1:10,x2:100,y2:100});
  assert.equal(yoloBoxes([box],0,800).length,0);
  assert.equal(yoloBoxes([{...box,confidence:Infinity}],1200,800).length,0);
  assert.equal(yoloBoxes([{...box,cascade_score:-0.2}],1200,800).length,0);
});
