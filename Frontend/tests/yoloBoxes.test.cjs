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
const {yoloBoxes, hybridFallbackBoxes, filterBoxesByDisease} = exported;
const detection = bbox => ({disease:'Cardiomegaly',confidence:0.9,cascade_score:0.6,bbox});
const hybridBox = bbox => ({disease:'Hernia',confidence:0.61,probability:0.96,source:'hybrid-gradcam',bbox});

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
test('hybridFallbackBoxes validates, clips and preserves valid hybrid boxes', () => {
  const [valid] = hybridFallbackBoxes([hybridBox({x1:-20,y1:300,x2:500,y2:900})],1200,800);
  assert.deepEqual(JSON.parse(JSON.stringify(valid.bbox)),{x1:0,y1:300,x2:500,y2:800});
  assert.equal(hybridFallbackBoxes([hybridBox({x1:NaN,y1:0,x2:20,y2:20})],1200,800).length,0);
  assert.equal(hybridFallbackBoxes([{...hybridBox({x1:10,y1:10,x2:100,y2:100}),probability:1.5}],1200,800).length,0);
});
test('filterBoxesByDisease only returns boxes for the selected disease', () => {
  const boxes = [
    {...hybridBox({x1:10,y1:10,x2:50,y2:50}), disease:'Atelectasis'},
    {...hybridBox({x1:60,y1:60,x2:120,y2:120}), disease:'Infiltration'},
    {...hybridBox({x1:20,y1:20,x2:80,y2:80}), disease:'Atelectasis'},
  ];
  assert.equal(filterBoxesByDisease(boxes, 'Atelectasis').length, 2);
  assert.equal(filterBoxesByDisease(boxes, 'infiltration').length, 1);
  assert.equal(filterBoxesByDisease(boxes, 'Effusion').length, 0);
  assert.equal(filterBoxesByDisease(boxes).length, 3);
});
