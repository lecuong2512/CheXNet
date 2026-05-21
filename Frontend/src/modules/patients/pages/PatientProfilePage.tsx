import React, { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate, Navigate } from 'react-router-dom';
import { message, Spin, Empty } from 'antd';
import {
    useGetPatientProfileQuery,
    useUploadScanMutation,
} from '../../../stores/baseApi';
import { formatAiModelLabel } from '../../../utils/aiModel';

// ── Scan type options (matches backend ScanType enum) ────────────────────────
import { CHEST_XRAY_SCAN_TYPES, DEFAULT_CHEST_SCAN_TYPE, CHEST_XRAY_SCOPE_NOTE } from '../../../constants/chestXray';

// ── Upload status type ────────────────────────────────────────────────────────
type UploadStep = 'idle' | 'uploading' | 'analyzing' | 'done' | 'error';

const PatientProfilePage: React.FC = () => {
    const { id } = useParams<{ id: string }>();
    const navigate = useNavigate();

    if (!id) {
        return <Navigate to="/" replace />;
    }

    const patientId = id;

    const { data: patient, isLoading, error, refetch } = useGetPatientProfileQuery(patientId);
    const [uploadScan] = useUploadScanMutation();

    // ── Scan comparison states ────────────────────────────────────────────────
    const [currentScanIndex, setCurrentScanIndex] = useState(0);
    const [previousScanIndex, setPreviousScanIndex] = useState(1);

    // ── Upload modal states ───────────────────────────────────────────────────
    const [showUploadModal, setShowUploadModal] = useState(false);
    const [isDragging, setIsDragging] = useState(false);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [scanType, setScanType] = useState(DEFAULT_CHEST_SCAN_TYPE);
    const [uploadStep, setUploadStep] = useState<UploadStep>('idle');
    const [uploadError, setUploadError] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const activeScan = useMemo(() => patient?.scans?.[currentScanIndex] || null, [patient, currentScanIndex]);
    const comparativeScan = useMemo(() =>
        patient?.scans?.[previousScanIndex] || patient?.scans?.[currentScanIndex + 1] || null,
        [patient, previousScanIndex, currentScanIndex]
    );
    const hasComparativeScan = Boolean(comparativeScan);

    useEffect(() => {
        setCurrentScanIndex(0);
        setPreviousScanIndex(1);
    }, [patientId]);

    // ── Cleanup preview URL on unmount ────────────────────────────────────────
    useEffect(() => {
        return () => {
            if (previewUrl) URL.revokeObjectURL(previewUrl);
        };
    }, [previewUrl]);

    // ── Drag & Drop handlers ──────────────────────────────────────────────────
    const handleFileDrop = useCallback((file: File) => {
        if (!file.type.startsWith('image/')) {
            message.error('Vui lòng chọn file ảnh (JPEG, PNG, DICOM).');
            return;
        }
        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        setUploadStep('idle');
        setUploadError(null);
    }, []);

    const onDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); };
    const onDragLeave = () => setIsDragging(false);
    const onDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files?.[0];
        if (file) handleFileDrop(file);
    };
    const onFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) handleFileDrop(file);
    };

    // ── Close & Reset modal ───────────────────────────────────────────────────
    const closeModal = () => {
        setShowUploadModal(false);
        setSelectedFile(null);
        setPreviewUrl(null);
        setScanType(DEFAULT_CHEST_SCAN_TYPE);
        setUploadStep('idle');
        setUploadError(null);
    };

    // ── Submit upload to backend ──────────────────────────────────────────────
    const handleSubmitUpload = async () => {
        if (!selectedFile) {
            message.warning('Vui lòng chọn file ảnh X-ray trước khi tải lên.');
            return;
        }

        setUploadStep('uploading');
        setUploadError(null);

        try {
            setUploadStep('analyzing');

            const result = await uploadScan({
                patientId,
                scanType,
                imageFile: selectedFile,
            }).unwrap();

            await refetch();
            setUploadStep('done');
            setCurrentScanIndex(0);

            const modelLabel = formatAiModelLabel(result?.aiModel);
            message.success(`Phân tích AI hoàn tất (${modelLabel})`);
            setTimeout(() => closeModal(), 1200);
        } catch (err: any) {
            setUploadStep('error');
            const errMsg = err?.data?.message || err?.data || 'Lỗi không xác định khi tải ảnh lên.';
            setUploadError(String(errMsg));
        }
    };

    // ── Loading / Error states ────────────────────────────────────────────────
    if (isLoading) {
        return (
            <div className="flex-1 flex items-center justify-center p-margin bg-background dark:bg-[#1a1d27] min-h-[60vh]">
                <Spin size="large" tip="Đang tải hồ sơ bệnh án..." />
            </div>
        );
    }

    if (error || !patient) {
        return (
            <div className="flex-1 flex flex-col items-center justify-center p-margin min-h-[60vh]">
                <Empty description="Không tìm thấy hồ sơ bệnh nhân!" />
                <button onClick={() => navigate('/patient-history')} className="mt-4 px-4 py-2 bg-primary text-white rounded-lg text-sm font-semibold">
                    Quay lại Lịch sử
                </button>
            </div>
        );
    }

    const vitals = patient.vitals || {
        heartRate: 75, bloodPressure: '120/80', spo2: 98, temperature: 36.8, lungIndex: 90,
    };

    return (
        <div className="flex-1 flex flex-col h-full overflow-hidden relative bg-medical-grid">

            {/* ── Upload X-ray Modal ───────────────────────────────────────── */}
            {showUploadModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
                    <div className="bg-white dark:bg-[#232736] rounded-2xl border border-outline-variant/50 shadow-2xl w-full max-w-lg flex flex-col overflow-hidden">
                        {/* Modal Header */}
                        <div className="flex items-center justify-between px-6 py-4 border-b border-outline-variant/30">
                            <div className="flex items-center gap-3">
                                <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center">
                                    <span className="material-symbols-outlined text-primary dark:text-[#2563eb] text-[20px]">upload</span>
                                </div>
                                <div>
                                    <h3 className="text-body-lg font-semibold text-on-surface dark:text-white m-0">Tải X-quang ngực</h3>
                                    <p className="text-xs text-on-surface-variant dark:text-gray-400 m-0 mt-0.5">{CHEST_XRAY_SCOPE_NOTE}</p>
                                    <p className="text-xs text-on-surface-variant dark:text-gray-400 m-0">Bệnh nhân: {patient.name} · {patientId}</p>
                                </div>
                            </div>
                            <button onClick={closeModal} disabled={uploadStep === 'uploading' || uploadStep === 'analyzing'}
                                className="p-1.5 rounded-full hover:bg-surface-container-high transition-colors disabled:opacity-40">
                                <span className="material-symbols-outlined text-[22px] text-on-surface-variant">close</span>
                            </button>
                        </div>

                        {/* Modal Body */}
                        <div className="p-6 flex flex-col gap-5">
                            {/* Scan Type Selector */}
                            <div>
                                <label className="block text-label-sm font-semibold text-on-surface dark:text-gray-200 mb-2">
                                    Loại phim chụp
                                </label>
                                <div className="flex flex-wrap gap-2">
                                    {CHEST_XRAY_SCAN_TYPES.map((type) => (
                                        <button key={type}
                                            onClick={() => setScanType(type)}
                                            className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                                                scanType === type
                                                    ? 'bg-primary text-white border-primary shadow-sm'
                                                    : 'bg-background dark:bg-[#1a1d27] text-on-surface-variant dark:text-gray-400 border-outline-variant/40 hover:border-primary/50'
                                            }`}
                                        >
                                            {type}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Drag & Drop Zone */}
                            <div
                                onDragOver={onDragOver}
                                onDragLeave={onDragLeave}
                                onDrop={onDrop}
                                onClick={() => fileInputRef.current?.click()}
                                className={`relative border-2 border-dashed rounded-xl flex flex-col items-center justify-center gap-3 transition-all cursor-pointer min-h-[200px] ${
                                    isDragging
                                        ? 'border-primary bg-primary/5 dark:bg-primary/10'
                                        : 'border-outline-variant/50 hover:border-primary/50 hover:bg-surface-container-high/50 dark:hover:bg-[#1a1d27]'
                                }`}
                            >
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    accept="image/*,.dcm"
                                    className="hidden"
                                    onChange={onFileInputChange}
                                />
                                {previewUrl ? (
                                    <div className="w-full h-full flex flex-col items-center gap-2 p-2">
                                        <img src={previewUrl} alt="preview" className="max-h-[160px] rounded-lg object-contain border border-outline-variant/30" />
                                        <p className="text-xs text-primary dark:text-[#2563eb] font-semibold truncate max-w-full px-2">
                                            {selectedFile?.name}
                                        </p>
                                        <span className="text-[11px] text-on-surface-variant dark:text-gray-400">
                                            Nhấp để chọn lại
                                        </span>
                                    </div>
                                ) : (
                                    <>
                                        <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center">
                                            <span className="material-symbols-outlined text-primary dark:text-[#2563eb] text-[32px]">add_photo_alternate</span>
                                        </div>
                                        <div className="text-center">
                                            <p className="text-sm font-semibold text-on-surface dark:text-white">
                                                Kéo thả hoặc nhấp để chọn ảnh
                                            </p>
                                            <p className="text-xs text-on-surface-variant dark:text-gray-400 mt-1">
                                                Hỗ trợ JPEG, PNG, DICOM · Tối đa 10MB
                                            </p>
                                        </div>
                                    </>
                                )}
                            </div>

                            {/* Upload Progress */}
                            {(uploadStep === 'uploading' || uploadStep === 'analyzing') && (
                                <div className="bg-primary/5 border border-primary/20 rounded-xl p-4 flex items-center gap-4">
                                    <Spin size="small" />
                                    <div>
                                        <p className="text-sm font-semibold text-primary dark:text-[#2563eb] m-0">
                                            {uploadStep === 'uploading' ? 'Đang tải lên máy chủ...' : '🧠 AI đang phân tích phim X-quang...'}
                                        </p>
                                        <p className="text-xs text-on-surface-variant dark:text-gray-400 m-0 mt-0.5">
                                            {uploadStep === 'uploading'
                                                ? 'Đang gửi dữ liệu lên hệ thống CheXNet V3'
                                                : 'CheXNet đang chạy inference (DenseNet-121 / ConvNeXtV2)...'
                                            }
                                        </p>
                                    </div>
                                </div>
                            )}

                            {uploadStep === 'done' && (
                                <div className="bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-700/50 rounded-xl p-4 flex items-center gap-3">
                                    <span className="material-symbols-outlined text-emerald-600 dark:text-emerald-400 text-[24px]">check_circle</span>
                                    <p className="text-sm font-semibold text-emerald-700 dark:text-emerald-300 m-0">Phân tích hoàn tất! Đang cập nhật hồ sơ...</p>
                                </div>
                            )}

                            {uploadStep === 'error' && uploadError && (
                                <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-700/50 rounded-xl p-4">
                                    <p className="text-sm font-semibold text-red-600 dark:text-red-400 m-0 flex items-center gap-2">
                                        <span className="material-symbols-outlined text-[18px]">error</span>
                                        Lỗi tải lên
                                    </p>
                                    <p className="text-xs text-red-500 dark:text-red-400 mt-1 m-0">{uploadError}</p>
                                </div>
                            )}
                        </div>

                        {/* Modal Footer */}
                        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-outline-variant/30 bg-surface-container-lowest dark:bg-[#1a1d27]/60">
                            <button onClick={closeModal}
                                disabled={uploadStep === 'uploading' || uploadStep === 'analyzing'}
                                className="px-4 py-2 rounded-lg text-sm font-semibold text-on-surface-variant dark:text-gray-400 hover:bg-surface-container-high transition-colors disabled:opacity-40">
                                Hủy
                            </button>
                            <button
                                onClick={handleSubmitUpload}
                                disabled={!selectedFile || uploadStep === 'uploading' || uploadStep === 'analyzing' || uploadStep === 'done'}
                                className="px-5 py-2 rounded-lg text-sm font-semibold bg-primary text-white hover:bg-primary/90 transition-all shadow-sm flex items-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
                            >
                                {uploadStep === 'uploading' || uploadStep === 'analyzing' ? (
                                    <><Spin size="small" /><span>Đang xử lý...</span></>
                                ) : (
                                    <><span className="material-symbols-outlined text-[18px]">cloud_upload</span><span>Gửi & Phân tích AI</span></>
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* ── Sticky Header ────────────────────────────────────────────── */}
            <header className="sticky top-0 z-30 bg-white/80 dark:bg-[#232736]/80 backdrop-blur-xl border-b border-outline-variant/30 px-gutter py-stack-md flex flex-wrap justify-between items-center gap-4 shadow-sm">
                <div className="flex items-center gap-stack-md">
                    <button onClick={() => navigate('/patient-history')}
                        className="p-1.5 hover:bg-surface-container-high rounded-full transition-colors flex items-center justify-center"
                        title="Quay lại danh sách">
                        <span className="material-symbols-outlined text-[24px]">arrow_back</span>
                    </button>
                    <div className="w-12 h-12 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center overflow-hidden">
                        <span className="material-symbols-outlined text-[28px] text-primary dark:text-[#2563eb]">person</span>
                    </div>
                    <div>
                        <div className="flex items-center gap-2">
                            <h2 className="text-headline-md font-headline-md text-on-surface dark:text-white m-0">{patient.name}</h2>
                            <span className="px-2.5 py-0.5 rounded-full bg-surface-container-highest dark:bg-gray-800 border border-outline-variant/50 text-label-sm font-label-sm text-on-surface-variant dark:text-gray-300 font-mono">
                                ID: {patient.id}
                            </span>
                        </div>
                        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-label-sm font-label-sm text-on-surface-variant dark:text-gray-400 mt-1">
                            <span>{patient.gender}, {patient.age} tuổi</span>
                            <span className="w-1 h-1 rounded-full bg-outline hidden sm:inline-block"></span>
                            <span>Nhóm máu: {patient.bloodType}</span>
                            <span className="w-1 h-1 rounded-full bg-outline hidden sm:inline-block"></span>
                            <span>Khoa: {patient.department || 'Nội Hô Hấp'}</span>
                        </div>
                    </div>
                </div>

                <div className="flex items-center gap-3">
                    {/* Upload CTA Button */}
                    <button
                        id="upload-xray-btn"
                        onClick={() => setShowUploadModal(true)}
                        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-white text-sm font-semibold shadow-sm hover:bg-primary/90 transition-all"
                    >
                        <span className="material-symbols-outlined text-[18px]">add_photo_alternate</span>
                        Tải X-quang ngực
                    </button>

                    {/* System Status */}
                    <div className="flex items-center gap-3 bg-white dark:bg-[#1a1d27] border border-outline-variant/30 rounded-lg px-4 py-2 shadow-sm">
                        <div className="flex items-center gap-2 text-label-bold font-label-bold text-on-surface dark:text-white text-xs sm:text-sm">
                            <span className="w-2.5 h-2.5 rounded-full bg-[#10b981] animate-pulse"></span>
                            Hệ thống: Ổn định
                        </div>
                        <div className="w-[1px] h-4 bg-outline-variant/50 mx-2"></div>
                        <button onClick={() => message.info('Đang xuất bản in báo cáo bệnh án...')}
                            className="text-primary dark:text-[#2563eb] hover:opacity-80 transition-opacity p-1 flex items-center justify-center"
                            title="In báo cáo bệnh án">
                            <span className="material-symbols-outlined text-[20px]">print</span>
                        </button>
                    </div>
                </div>
            </header>

            {/* ── Main Content ─────────────────────────────────────────────── */}
            <div className="flex-1 overflow-y-auto p-gutter pb-margin">
                <div className="max-w-container-max mx-auto space-y-gutter">

                    {/* Vitals & AI Summary */}
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-gutter">
                        {/* Vital Signs */}
                        <div className="lg:col-span-5 bg-white dark:bg-[#232736] rounded-xl border border-outline-variant/50 shadow-sm p-stack-md flex flex-col transition-all duration-300">
                            <div className="flex justify-between items-center mb-stack-md">
                                <h3 className="text-body-lg font-body-lg font-semibold flex items-center gap-2 text-on-surface dark:text-white">
                                    <span className="material-symbols-outlined text-primary dark:text-[#2563eb]">vital_signs</span>
                                    Dấu hiệu Sinh tồn
                                </h3>
                                <span className="text-[11px] text-on-surface-variant dark:text-gray-400">Cập nhật 5 phút trước</span>
                            </div>
                            <div className="grid grid-cols-2 gap-stack-sm flex-1">
                                <div className="bg-background dark:bg-[#1a1d27] border border-outline-variant/30 rounded-lg p-3 flex flex-col justify-between">
                                    <div className="text-label-sm font-label-sm text-on-surface-variant dark:text-gray-400 mb-1">Nhịp tim</div>
                                    <div className="flex items-baseline gap-1">
                                        <span className="text-headline-md font-headline-md text-on-surface dark:text-white font-bold">{vitals.heartRate}</span>
                                        <span className="text-label-sm font-label-sm text-on-surface-variant dark:text-gray-400">bpm</span>
                                    </div>
                                </div>
                                <div className="bg-background dark:bg-[#1a1d27] border border-outline-variant/30 rounded-lg p-3 flex flex-col justify-between">
                                    <div className="text-label-sm font-label-sm text-on-surface-variant dark:text-gray-400 mb-1">Huyết áp</div>
                                    <div className="flex items-baseline gap-1">
                                        <span className="text-headline-md font-headline-md text-on-surface dark:text-white font-bold">{vitals.bloodPressure}</span>
                                        <span className="text-label-sm font-label-sm text-on-surface-variant dark:text-gray-400">mmHg</span>
                                    </div>
                                </div>
                                <div className={`border rounded-lg p-3 flex flex-col justify-between relative overflow-hidden ${vitals.spo2 < 93 ? 'bg-error-container/20 border-error/40' : 'bg-background dark:bg-[#1a1d27] border-outline-variant/30'}`}>
                                    {vitals.spo2 < 93 && <div className="absolute top-0 right-0 w-8 h-8 bg-error/10 rounded-bl-xl"></div>}
                                    <div className={`text-label-sm font-label-sm font-semibold mb-1 flex items-center gap-1 ${vitals.spo2 < 93 ? 'text-on-error-container' : 'text-on-surface-variant dark:text-gray-400'}`}>
                                        {vitals.spo2 < 93 && <span className="material-symbols-outlined text-[14px]">warning</span>}
                                        SpO2
                                    </div>
                                    <div className="flex items-baseline gap-1">
                                        <span className={`text-headline-md font-headline-md font-bold ${vitals.spo2 < 93 ? 'text-error' : 'text-on-surface dark:text-white'}`}>{vitals.spo2}</span>
                                        <span className={`text-label-sm font-label-sm ${vitals.spo2 < 93 ? 'text-error' : 'text-on-surface-variant dark:text-gray-400'}`}>%</span>
                                    </div>
                                </div>
                                <div className="bg-background dark:bg-[#1a1d27] border border-outline-variant/30 rounded-lg p-3 flex flex-col justify-between">
                                    <div className="text-label-sm font-label-sm text-on-surface-variant dark:text-gray-400 mb-1">Nhiệt độ</div>
                                    <div className="flex items-baseline gap-1">
                                        <span className="text-headline-md font-headline-md text-on-surface dark:text-white font-bold">{vitals.temperature}</span>
                                        <span className="text-label-sm font-label-sm text-on-surface-variant dark:text-gray-400">°C</span>
                                    </div>
                                </div>
                                <div className="bg-primary/5 dark:bg-primary-container/10 border border-primary/20 rounded-lg p-3 flex flex-col justify-between col-span-2">
                                    <div className="text-label-sm font-label-sm text-primary dark:text-[#2563eb] font-semibold mb-1">Chỉ số Phổi (Lung Index)</div>
                                    <div className="flex items-baseline gap-1">
                                        <span className="text-headline-md font-headline-md text-primary dark:text-[#2563eb] font-bold">{vitals.lungIndex}</span>
                                        <span className="text-label-sm font-label-sm text-primary dark:text-[#2563eb]">LHI</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* AI Clinical Alert Summary */}
                        <div className="lg:col-span-7 bg-white dark:bg-[#232736] rounded-xl border border-outline-variant/50 shadow-sm p-stack-md flex flex-col relative overflow-hidden transition-all duration-300">
                            <div className="absolute left-0 top-0 bottom-0 w-1 bg-tertiary"></div>
                            <div className="flex justify-between items-start mb-stack-md">
                                <div>
                                    <h3 className="text-body-lg font-body-lg font-semibold flex items-center gap-2 mb-1 text-on-surface dark:text-white">
                                        <span className="material-symbols-outlined text-tertiary">notifications_active</span>
                                        Tóm tắt Bệnh lý Phổi (AI Summary)
                                    </h3>
                                    <p className="text-label-sm font-label-sm text-on-surface-variant dark:text-gray-400">
                                        Phân tích từ lần chụp gần nhất ({activeScan?.date || 'Không rõ'})
                                    </p>
                                </div>
                                <span className="px-3 py-1 bg-tertiary-container/10 text-tertiary-container dark:text-orange-300 border border-tertiary/20 rounded-md text-label-sm font-label-sm font-bold">
                                    Mức độ: Cần theo dõi
                                </span>
                            </div>
                            <div className="bg-background dark:bg-[#1a1d27] border border-outline-variant/30 rounded-lg p-4 flex-1">
                                <p className="text-body-md font-body-md text-on-surface dark:text-gray-200 leading-relaxed">
                                    {activeScan?.description || 'Mô hình AI chưa phát hiện các dấu hiệu bệnh lý nguy hiểm trên phim phổi gần nhất. Bệnh nhân tiếp tục lịch theo dõi lâm sàng định kỳ theo đề xuất của bác sĩ điều trị.'}
                                </p>
                                {activeScan && activeScan.predictions?.length > 0 && (
                                    <div className="mt-4 flex flex-wrap gap-2">
                                        {activeScan.predictions.map((p: any, idx: number) => (
                                            <span key={idx} className="px-2.5 py-1 bg-surface-container-high dark:bg-gray-800 rounded text-xs font-semibold text-on-surface-variant dark:text-gray-300 border border-outline-variant/30">
                                                #{p.name} ({p.probability}%)
                                            </span>
                                        ))}
                                        {vitals.spo2 < 93 && (
                                            <span className="px-2.5 py-1 bg-error/10 text-error rounded text-xs font-bold border border-error/20">
                                                SpO2 Giảm
                                            </span>
                                        )}
                                    </div>
                                )}

                                {/* Upload CTA when no scans exist */}
                                {(!patient.scans || patient.scans.length === 0) && (
                                    <div className="mt-4 flex flex-col items-center gap-3 py-4 border border-dashed border-outline-variant/50 rounded-lg">
                                        <span className="material-symbols-outlined text-[40px] text-on-surface-variant/40">add_photo_alternate</span>
                                        <p className="text-sm text-on-surface-variant dark:text-gray-400 text-center m-0">
                                            Chưa có phim chụp X-quang nào.<br />Tải lên phim để bắt đầu phân tích AI.
                                        </p>
                                        <button
                                            onClick={() => setShowUploadModal(true)}
                                            className="px-4 py-2 rounded-lg bg-primary text-white text-sm font-semibold hover:bg-primary/90 transition-all flex items-center gap-2"
                                        >
                                            <span className="material-symbols-outlined text-[16px]">upload</span>
                                            Tải lên ngay
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Comparative View */}
                    <div className="bg-white dark:bg-[#232736] rounded-xl border border-outline-variant/50 shadow-sm p-stack-md transition-all duration-300">
                        <div className="flex flex-wrap justify-between items-center gap-3 mb-stack-md border-b border-outline-variant/30 pb-3">
                            <h3 className="text-body-lg font-body-lg font-semibold flex items-center gap-2 text-on-surface dark:text-white">
                                <span className="material-symbols-outlined text-primary dark:text-[#2563eb]">compare</span>
                                Trạm Phân tích So sánh
                            </h3>
                            <span className="text-label-sm font-label-sm text-primary dark:text-[#2563eb] px-2">
                                Chỉ X-quang ngực
                            </span>
                        </div>
                        <div className={`grid grid-cols-1 gap-stack-md ${hasComparativeScan ? 'md:grid-cols-2' : 'max-w-xl mx-auto'}`}>
                            {/* Previous Scan — chỉ hiện khi có ít nhất 2 phim để so sánh */}
                            {hasComparativeScan && comparativeScan && (
                                <div className="flex flex-col border border-outline-variant/50 rounded-lg overflow-hidden bg-background dark:bg-[#1a1d27]">
                                    <div className="bg-surface-container-high dark:bg-gray-800 px-3 py-2 flex justify-between items-center border-b border-outline-variant/50">
                                        <span className="text-label-bold font-label-bold text-on-surface dark:text-white">Phim tham chiếu cũ</span>
                                        <span className="text-label-sm font-label-sm text-on-surface-variant dark:text-gray-400 font-mono">{comparativeScan.date}</span>
                                    </div>
                                    <div className="relative aspect-[4/3] w-full bg-black flex items-center justify-center p-2">
                                        <img alt="Previous lungs radiograph" className="h-full object-contain max-h-[300px] border border-gray-900 rounded"
                                            src={comparativeScan.image} />
                                        <div className="absolute bottom-2 left-2 bg-black/60 text-white text-[10px] px-2 py-0.5 rounded font-mono">
                                            {comparativeScan.type}
                                        </div>
                                    </div>
                                    <div className="p-3 border-t border-outline-variant/30">
                                        <p className="text-xs text-on-surface-variant dark:text-gray-400 italic line-clamp-2">
                                            {comparativeScan.description}
                                        </p>
                                    </div>
                                </div>
                            )}

                            {/* Current Scan */}
                            <div className="flex flex-col border border-primary/50 dark:border-primary-container rounded-lg overflow-hidden bg-background dark:bg-[#1a1d27] shadow-[0_0_15px_rgba(0,74,198,0.05)] relative">
                                <div className="absolute top-0 right-0 w-10 h-10 bg-primary/10 rounded-bl-2xl z-10 flex items-center justify-center border-b border-l border-primary/20">
                                    <span className="material-symbols-outlined text-primary dark:text-[#2563eb] text-[18px]">new_releases</span>
                                </div>
                                <div className="bg-primary/5 dark:bg-primary-container/10 px-3 py-2 flex justify-between items-center border-b border-primary/20">
                                    <span className="text-label-bold font-label-bold text-primary dark:text-[#2563eb] font-semibold">Phim được chọn phân tích</span>
                                    <span className="text-label-sm font-label-sm text-primary dark:text-[#2563eb] font-mono">{activeScan?.date || '—'}</span>
                                </div>
                                <div className="relative aspect-[4/3] w-full bg-black flex items-center justify-center p-2">
                                    {activeScan ? (
                                        <>
                                            <img alt="Current lungs radiograph" className="h-full object-contain max-h-[300px] border border-gray-900 rounded"
                                                src={activeScan.image} />
                                            <div className="absolute bottom-2 left-2 bg-black/60 text-white text-[10px] px-2 py-0.5 rounded font-mono">
                                                {activeScan.type}
                                            </div>
                                            {activeScan.predictions?.some((p: any) => p.probability > 75) && (
                                                <div className="absolute bottom-1/4 right-1/4 w-1/3 h-1/4 border-[2px] border-tertiary border-dashed rounded-sm bg-tertiary/10 flex items-start justify-end p-1">
                                                    <span className="text-[9px] bg-tertiary text-white px-1 rounded-sm font-bold uppercase">
                                                        AI Bất thường (+{activeScan.predictions[0].probability}%)
                                                    </span>
                                                </div>
                                            )}
                                        </>
                                    ) : (
                                        <div className="flex flex-col items-center gap-3 text-center px-4">
                                            <span className="material-symbols-outlined text-[48px] text-on-surface-variant/30">radiology</span>
                                            <p className="text-sm text-on-surface-variant dark:text-gray-400 m-0">
                                                Chưa có phim X-quang.<br />Tải lên phim để bắt đầu phân tích AI.
                                            </p>
                                            <button
                                                onClick={() => setShowUploadModal(true)}
                                                className="px-4 py-2 rounded-lg bg-primary text-white text-xs font-semibold hover:bg-primary/90 transition-all flex items-center gap-2"
                                            >
                                                <span className="material-symbols-outlined text-[16px]">upload</span>
                                                Tải X-quang ngực
                                            </button>
                                        </div>
                                    )}
                                </div>
                                <div className="p-3 border-t border-outline-variant/30 flex justify-between items-center">
                                    <span className="text-xs text-on-surface-variant dark:text-gray-400 flex items-center gap-1">
                                        <span className="material-symbols-outlined text-[14px]">visibility</span>
                                        {activeScan ? 'Chế độ xem: AI Bounding Box Overlay' : 'Chưa có dữ liệu phân tích'}
                                    </span>
                                    {activeScan && (
                                        <button onClick={() => navigate('/')} className="text-primary dark:text-[#2563eb] text-xs font-bold hover:underline">
                                            Mở Diagnostic Hub
                                        </button>
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Timeline Archive */}
                    <div className="bg-white dark:bg-[#232736] rounded-xl border border-outline-variant/50 shadow-sm p-stack-md transition-all duration-300">
                        <div className="flex justify-between items-center mb-stack-md">
                            <h3 className="text-body-lg font-body-lg font-semibold flex items-center gap-2 text-on-surface dark:text-white">
                                <span className="material-symbols-outlined text-primary dark:text-[#2563eb]">timeline</span>
                                Lưu trữ Lịch sử Hình ảnh Hô hấp (Timeline)
                            </h3>
                            <button
                                onClick={() => setShowUploadModal(true)}
                                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-primary/30 text-primary dark:text-[#2563eb] text-xs font-semibold hover:bg-primary/5 transition-colors"
                            >
                                <span className="material-symbols-outlined text-[15px]">add</span>
                                Thêm phim mới
                            </button>
                        </div>
                        <div className="overflow-x-auto timeline-scroll pb-2">
                            <div className="flex gap-4 min-w-max">
                                {patient.scans?.map((scan: any, idx: number) => {
                                    const hasAnomaly = scan.predictions && scan.predictions.length > 0;
                                    const isSelected = idx === currentScanIndex;
                                    return (
                                        <div key={scan.id} onClick={() => setCurrentScanIndex(idx)}
                                            className={`w-64 border rounded-lg p-3 bg-background dark:bg-[#1a1d27] transition-all cursor-pointer ${
                                                isSelected
                                                    ? 'border-primary dark:border-[#2563eb] ring-1 ring-primary dark:ring-[#2563eb] translate-y-[-2px] shadow-md'
                                                    : 'border-outline-variant/40 hover:border-primary/50'
                                            }`}>
                                            <div className="flex justify-between items-start mb-2">
                                                <span className="text-label-bold font-label-bold text-on-surface dark:text-white font-mono">{scan.date}</span>
                                                <span className={`material-symbols-outlined text-[18px] ${isSelected ? 'text-primary dark:text-[#2563eb] icon-fill' : 'text-outline'}`}>
                                                    pulmonology
                                                </span>
                                            </div>
                                            <p className="text-xs text-on-surface-variant dark:text-gray-400 mb-2 truncate" title={scan.type}>{scan.type}</p>
                                            <div className="flex items-center gap-2">
                                                <span className={`w-2 h-2 rounded-full ${hasAnomaly ? 'bg-tertiary' : 'bg-emerald-500'}`}></span>
                                                <span className="text-[11px] font-medium text-on-surface dark:text-gray-200">
                                                    {hasAnomaly ? `Phát hiện: ${scan.predictions[0].name}` : 'Ổn định / Bình thường'}
                                                </span>
                                            </div>
                                        </div>
                                    );
                                })}

                                {/* Add new scan card */}
                                <div onClick={() => setShowUploadModal(true)}
                                    className="w-48 border border-outline-variant/40 rounded-lg p-3 bg-background dark:bg-[#1a1d27] hover:border-primary/50 transition-colors cursor-pointer border-dashed flex flex-col items-center justify-center gap-2 text-center">
                                    <span className="material-symbols-outlined text-[28px] text-primary/40 dark:text-[#2563eb]/40">add_photo_alternate</span>
                                    <span className="text-xs text-primary dark:text-[#2563eb] font-bold">Tải lên phim mới</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PatientProfilePage;
