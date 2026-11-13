// Run after building Backend: node --test tests/upload-consistency.test.cjs
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { randomUUID } = require('node:crypto');
const mongoose = require('mongoose');
const { ScanService } = require('../dist/modules/scans/scan.service');
const { DiagnosisService } = require('../dist/modules/diagnoses/diagnosis.service');
const { ResearchService } = require('../dist/modules/research/research.service');
const { PatientService } = require('../dist/modules/patients/patient.service');
const { saveReferenceCase } = require('../dist/modules/diagnoses/diagnosis.reference');
const axios = require('axios');
const Scan = require('../dist/modules/scans/scan.model').default;
const Diagnosis = require('../dist/modules/diagnoses/diagnosis.model').default;
const Patient = require('../dist/modules/patients/patient.model').default;
const { socketService } = require('../dist/socket/socket.service');

const dbName = 'chexnet_upload_test_' + randomUUID().replaceAll('-', '');
const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'chexnet-upload-test-'));
const service = new ScanService();
const userId = new mongoose.Types.ObjectId().toString();
let patientId;
let calls = 0;

function file(content = 'fixture') {
    const filename = randomUUID() + '.png';
    const filePath = path.join(directory, filename);
    fs.writeFileSync(filePath, content);
    return { path: filePath, filename };
}

before(async () => {
    await mongoose.connect('mongodb://127.0.0.1:27017/' + dbName);
    await Scan.init();
    // Only this isolated test uses a raw fixture patient.
    const patient = await Patient.collection.insertOne({
        patientCode: 'CONSISTENCY-TEST', isActive: true,
    });
    patientId = patient.insertedId.toString();
    service.callAIService = async () => {
        calls++;
        return {
            predictions: [{ className: 'Pneumonia', probability: 0.8 }],
            aiModel: 'fixture', processingTime: 1,
            hybridAnalysis: { predictions: [{ disease: 'Pneumonia', probability: 0.8 }] },
            inferenceMeta: { detectorStatus: 'unavailable', contractVersion: 1 },
        };
    };
});

after(async () => {
    assert.equal(mongoose.connection.name, dbName);
    await mongoose.connection.dropDatabase();
    await mongoose.disconnect();
    fs.rmSync(directory, { recursive: true, force: true });
});

test('empty research database has no invented timing, alerts or anatomical grid', async () => {
    const research = new ResearchService();
    const stats = await research.getStats();
    assert.equal(stats.data.totalScans, 0);
    assert.equal(stats.data.avgProcessingTime, null);
    assert.deepEqual(stats.data.clinicalAlerts, []);
    assert.deepEqual((await research.getHeatmap()).data, []);
    const trends = await research.getTrends();
    assert.equal(trends.data.length, 8);
    assert.ok(trends.data.every(week => week.data.length === 0));
});

test('repeated request returns the same committed result without new inference', async () => {
    const key = randomUUID();
    const firstFile = file();
    const first = await service.uploadAndProcess(patientId, userId, firstFile, { requestKey: key });
    const beforeCalls = calls;
    const secondFile = file();
    const repeated = await service.uploadAndProcess(patientId, userId, secondFile, { requestKey: key });
    assert.equal(String(repeated.data._id), String(first.data._id));
    assert.equal(calls, beforeCalls);
    assert.equal(await Scan.countDocuments({ requestKey: key }), 1);
    assert.equal(fs.existsSync(secondFile.path), false);
    assert.equal(fs.existsSync(firstFile.path), true);
    await assert.rejects(service.uploadAndProcess(patientId, userId, file('changed'), { requestKey: key }),
        error => error.statusCode === 409);
});

test('diagnosis write failure hides scan and removes only its own file', async () => {
    const original = Diagnosis.create;
    Diagnosis.create = async () => { throw new Error('injected diagnosis write failure'); };
    const uploaded = file();
    const requestKey = randomUUID();
    try {
        await assert.rejects(service.uploadAndProcess(patientId, userId, uploaded, { requestKey }));
    } finally {
        Diagnosis.create = original;
    }
    const scan = await Scan.findOne({ requestKey });
    assert.equal(scan.status, 'failed');
    assert.equal(await Diagnosis.countDocuments({ scanId: scan._id }), 0);
    assert.equal(fs.existsSync(uploaded.path), false);
    const listed = await service.getByPatient({ patientId });
    assert.ok(listed.data.every(item => item.status === 'done'));
});

test('concurrent duplicate is rejected while the first request owns inference', async () => {
    const original = service.callAIService;
    let release;
    let markStarted;
    const started = new Promise(resolve => { markStarted = resolve; });
    const held = new Promise(resolve => { release = resolve; });
    service.callAIService = async (...args) => {
        markStarted();
        await held;
        return original(...args);
    };
    const requestKey = randomUUID();
    const uploaded = file();
    const first = service.uploadAndProcess(patientId, userId, uploaded, { requestKey });
    await started;
    try {
        await assert.rejects(
            service.uploadAndProcess(patientId, userId, file(), { requestKey }),
            error => error.statusCode === 409,
        );
    } finally {
        release();
        service.callAIService = original;
    }
    const result = await first;
    assert.equal(result.data.status, 'done');
    assert.equal(await Scan.countDocuments({ requestKey }), 1);
});

test('notification failure preserves a completed scan and file', async () => {
    const original = socketService.sendToRoom;
    socketService.sendToRoom = () => { throw new Error('injected socket failure'); };
    const uploaded = file();
    try {
        const result = await service.uploadAndProcess(patientId, userId, uploaded, { requestKey: randomUUID() });
        assert.equal(result.data.status, 'done');
        assert.equal(fs.existsSync(uploaded.path), true);
        assert.equal(await Diagnosis.countDocuments({ scanId: result.data._id }), 1);
    } finally {
        socketService.sendToRoom = original;
    }
});

test('startup completes a scan with diagnosis and fails a partial scan', async () => {
    const completeFile = file();
    const partialFile = file();
    const make = uploaded => Scan.create({
        patientId, uploadedBy: userId, imagePath: uploaded.path, imageUrl: '/fixture',
        status: 'processing',
    });
    const complete = await make(completeFile);
    const partial = await make(partialFile);
    const diagnosis = await Diagnosis.create({ scanId: complete._id, patientId });
    const reader = new DiagnosisService();
    await assert.rejects(reader.getById(String(diagnosis._id)), error => error.statusCode === 404);
    const waiting = await reader.getByPatient({ patientId });
    assert.ok(waiting.data.every(item => String(item._id) !== String(diagnosis._id)));
    await service.recoverInterruptedUploads();
    assert.equal((await Scan.findById(complete._id)).status, 'done');
    assert.equal((await Scan.findById(partial._id)).status, 'failed');
    assert.equal(fs.existsSync(completeFile.path), true);
    assert.equal(fs.existsSync(partialFile.path), false);
    assert.equal(String((await reader.getById(String(diagnosis._id))).data._id), String(diagnosis._id));
});

test('doctor verification atomically rejects stale and concurrent edits without changing AI findings', async () => {
    const result = await service.uploadAndProcess(patientId, userId, file(), { requestKey: randomUUID() });
    const id = String(result.data.diagnosis._id);
    const reader = new DiagnosisService();
    const first = await reader.verify(id, 'verified', 'First saved conclusion', userId, 0);
    assert.equal(first.data.revision, 1);
    await assert.rejects(reader.verify(id, 'flagged', 'stale text', userId, 0), error => error.statusCode === 409);
    const competing = await Promise.allSettled([
        reader.verify(id, 'verified', 'Second editor', userId, 1),
        reader.verify(id, 'flagged', 'Third editor', userId, 1),
    ]);
    assert.equal(competing.filter(item => item.status === 'fulfilled').length, 1);
    assert.equal(competing.find(item => item.status === 'rejected').reason.statusCode, 409);
    const stored = await Diagnosis.findById(id);
    assert.equal(stored.revision, 2);
    assert.deepEqual(stored.hybridAnalysis, result.data.diagnosis.hybridAnalysis);
});

test('research counts detected diseases once per completed scan and reports legacy missing data', async () => {
    const research = new ResearchService();
    const beforeStats = (await research.getStats()).data;
    const beforeTrends = (await research.getTrends()).data;
    const sum = (rows, disease) => rows.reduce((total, week) =>
        total + week.data.filter(item => item._id === disease).reduce((count, item) => count + item.count, 0), 0);
    const predictions = [
        { disease: 'Pneumonia', probability: 0.8, lesion_location: 'Phổi phải, vùng dưới' },
        { disease: 'Pneumonia', probability: 0.8, lesion_location: 'Phổi phải, vùng dưới' },
        { disease: 'Effusion', probability: 0.7, lesion_location: 'Phổi phải, vùng dưới' },
    ];
    async function fixture(status, analysis, createdAt) {
        const uploaded = file();
        const scan = await Scan.create({ patientId, uploadedBy: userId, imagePath: uploaded.path,
            imageUrl: '/fixture', status });
        const diagnosis = await Diagnosis.create({
            scanId: scan._id, patientId, processingTime: 2,
            predictions: ['Pneumonia', 'Effusion', 'Mass', 'No Finding'].map(className => ({ className, probability: 0.95 })),
            hybridAnalysis: analysis,
        });
        if (createdAt) await Diagnosis.collection.updateOne({ _id: diagnosis._id }, { $set: { createdAt } });
    }
    await fixture('done', { predictions });
    await fixture('done', { predictions: [] });
    await fixture('done', undefined);
    await fixture('processing', { predictions });
    await fixture('done', { predictions }, new Date('2000-01-01'));
    const stats = (await research.getStats()).data;
    assert.equal(stats.totalScans, beforeStats.totalScans + 3);
    assert.equal(stats.missingDetectionData, beforeStats.missingDetectionData + 1);
    const afterTrends = (await research.getTrends()).data;
    assert.equal(sum(afterTrends, 'Pneumonia'), sum(beforeTrends, 'Pneumonia') + 1);
    assert.equal(sum(afterTrends, 'Effusion'), sum(beforeTrends, 'Effusion') + 1);
    assert.equal(sum(afterTrends, 'Mass'), 0);
    assert.equal(sum(afterTrends, 'No Finding'), 0);
    const locations = (await research.getHeatmap()).data;
    assert.equal(locations.find(row => row.disease === 'Pneumonia').count, 1);
    assert.equal(locations.find(row => row.disease === 'Effusion').count, 1);
});

test('summary profiles omit heatmaps and return diagnoses for all ten displayed scans', async () => {
    const ids = [];
    for (let i = 0; i < 11; i++) {
        const uploaded = file();
        const scan = await Scan.create({ patientId, uploadedBy: userId, imagePath: uploaded.path,
            imageUrl: '/fixture', status: 'done' });
        const diagnosis = await Diagnosis.create({ scanId: scan._id, patientId,
            hybridAnalysis: { predictions: [{ disease: 'Effusion', probability: 0.9, heatmap: 'image-bytes' }] } });
        ids.push(String(diagnosis._id));
    }
    const service = new PatientService();
    const full = (await service.getById(patientId)).data;
    const summary = (await service.getById(patientId, true)).data;
    assert.equal(summary.scans.length, 10);
    assert.equal(summary.diagnoses.length, 10);
    for (const diagnosis of summary.diagnoses) {
        assert.ok(summary.scans.some(scan => String(scan._id) === String(diagnosis.scanId)));
        assert.equal(diagnosis.hybridAnalysis.predictions[0].heatmap, undefined);
    }
    assert.equal(full.diagnoses[0].hybridAnalysis.predictions[0].heatmap, 'image-bytes');
    assert.ok(JSON.stringify(summary).length < JSON.stringify(full).length);
});

test('reference retry reconciles pending metadata after a lost AI response', { timeout: 15000 }, async () => {
    const result = await service.uploadAndProcess(patientId, userId, file(), { requestKey: randomUUID() });
    const id = String(result.data.diagnosis._id);
    await assert.rejects(saveReferenceCase(id, ['Pneumonia'], userId), error => error.statusCode === 409);
    await new DiagnosisService().verify(id, 'verified', 'Reference fixture', userId, 0);
    let first = true;
    const original = axios.post;
    axios.post = async (_url, form) => {
        await new Promise((resolve, reject) => {
            form.on('end', resolve);
            form.on('error', reject);
            form.resume();
        });
        if (first) { first = false; throw new Error('controlled lost response after publication'); }
        return { data: { success: true, referenceId: 'ref_' + id, replayed: true } };
    };
    try {
        await assert.rejects(saveReferenceCase(id, ['Pneumonia'], userId), error => error.statusCode === 503);
        const pending = await Diagnosis.findById(id);
        assert.equal(pending.referenceCase.status, 'pending');
        assert.equal(pending.referenceCase.revisionAtSave, 1);
        await assert.rejects(saveReferenceCase(id, ['Effusion'], userId), error => error.statusCode === 409);
        await saveReferenceCase(id, ['Pneumonia'], userId);
        const ready = await Diagnosis.findById(id);
        assert.equal(ready.referenceCase.status, 'ready');
        assert.equal(ready.referenceCase.savedBy, userId);
        const replay = await saveReferenceCase(id, ['Pneumonia'], userId);
        assert.equal(replay.replayed, true);
    } finally {
        axios.post = original;
    }
});
