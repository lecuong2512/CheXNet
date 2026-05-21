import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { message } from 'antd';
import { useCreatePatientMutation, useUploadScanMutation } from '../../stores/baseApi';
import { formatAiModelLabel } from '../../utils/aiModel';

interface AddPatientModalProps {
  isOpen: boolean;
  onClose: () => void;
}

type StepType = 1 | 2 | 3;

import { CHEST_XRAY_SCAN_TYPES, DEFAULT_CHEST_SCAN_TYPE, CHEST_XRAY_SCOPE_NOTE } from '../../constants/chestXray';

export const AddPatientModal: React.FC<AddPatientModalProps> = ({ isOpen, onClose }) => {
  const navigate = useNavigate();
  const [createPatient, { isLoading: isCreating }] = useCreatePatientMutation();
  const [uploadScan, { isLoading: isUploading }] = useUploadScanMutation();

  // ── Flow State ─────────────────────────────────────────────────────────────
  const [step, setStep] = useState<StepType>(1);

  // ── Step 1: Patient Form State ─────────────────────────────────────────────
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState<'Nam' | 'Nữ' | 'Khác'>('Nam');
  const [bloodType, setBloodType] = useState('O');
  const [department, setDepartment] = useState('Khoa Nội tổng hợp');
  const [phone, setPhone] = useState('');
  const [address, setAddress] = useState('');
  
  // Vitals State
  const [showVitals, setShowVitals] = useState(false);
  const [heartRate, setHeartRate] = useState('75');
  const [bloodPressure, setBloodPressure] = useState('120/80');
  const [spo2, setSpo2] = useState('98');
  const [temperature, setTemperature] = useState('36.8');
  const [lungIndex, setLungIndex] = useState('90');

  // ── Step 2: Upload Scan State ──────────────────────────────────────────────
  const [newPatient, setNewPatient] = useState<any>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [scanType, setScanType] = useState(DEFAULT_CHEST_SCAN_TYPE);
  const [isDragging, setIsDragging] = useState(false);

  // ── Step 3: Progressive AI Status State ─────────────────────────────────────
  const [progressPercent, setProgressPercent] = useState(0);
  const [progressMessage, setProgressMessage] = useState('Đang khởi tạo...');
  const [activeModelLabel, setActiveModelLabel] = useState('CheXNet');

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Reset modal when closing/opening
  useEffect(() => {
    if (isOpen) {
      setStep(1);
      setName('');
      setAge('');
      setGender('Nam');
      setBloodType('O');
      setDepartment('Khoa Nội tổng hợp');
      setPhone('');
      setAddress('');
      setNewPatient(null);
      setSelectedFile(null);
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        setPreviewUrl(null);
      }
      setScanType(DEFAULT_CHEST_SCAN_TYPE);
      setShowVitals(false);
      setHeartRate('75');
      setBloodPressure('120/80');
      setSpo2('98');
      setTemperature('36.8');
      setLungIndex('90');
    }
  }, [isOpen]);

  // Clean up URL object
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  // ── Drag & Drop Handlers ───────────────────────────────────────────────────
  const handleFileDrop = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) {
      message.error('Vui lòng chọn file ảnh (JPEG, PNG, DICOM).');
      return;
    }
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  }, []);

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => {
    setIsDragging(false);
  };

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

  // ── Action Handlers ────────────────────────────────────────────────────────
  const handleSavePatient = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || name.trim().length < 2) {
      message.warning('Tên bệnh nhân phải có tối thiểu 2 ký tự.');
      return;
    }
    const parsedAge = parseInt(age);
    if (isNaN(parsedAge) || parsedAge < 0 || parsedAge > 150) {
      message.warning('Tuổi bệnh nhân phải nằm trong khoảng 0 - 150.');
      return;
    }

    const payload: any = {
      name: name.trim(),
      age: parsedAge,
      gender,
      bloodType: bloodType.trim() || undefined,
      department: department.trim() || undefined,
      phone: phone.trim() || undefined,
      address: address.trim() || undefined,
    };

    if (showVitals) {
      payload.vitals = {
        heartRate: heartRate ? parseInt(heartRate) : undefined,
        bloodPressure: bloodPressure.trim() || undefined,
        spo2: spo2 ? parseInt(spo2) : undefined,
        temperature: temperature ? parseFloat(temperature) : undefined,
        lungIndex: lungIndex ? parseInt(lungIndex) : undefined,
      };
    }

    try {
      const res = await createPatient(payload).unwrap();
      if (res.success && res.data) {
        setNewPatient(res.data);
        message.success('Tạo hồ sơ bệnh nhân thành công!');
        setStep(2); // Move to upload step
      }
    } catch (err: any) {
      const errMsg = err?.data?.message || err?.data || 'Không thể tạo bệnh nhân mới.';
      message.error(`Lỗi: ${errMsg}`);
    }
  };

  const handleStartAnalysis = async () => {
    if (!selectedFile) {
      message.warning('Vui lòng chọn hoặc kéo thả phim chụp X-quang.');
      return;
    }

    setStep(3);
    setProgressPercent(5);
    setProgressMessage('Đang tải ảnh lên máy chủ...');
    setActiveModelLabel('CheXNet');

    const progressInterval = setInterval(() => {
      setProgressPercent((prev) => (prev < 88 ? prev + 2 : prev));
    }, 800);

    try {
      setProgressPercent(15);
      setProgressMessage('Đang gửi ảnh X-quang tới Backend...');

      const result = await uploadScan({
        patientId: newPatient.patientCode,
        scanType,
        imageFile: selectedFile,
      }).unwrap();

      clearInterval(progressInterval);

      const modelVersion = result?.aiModel as string | undefined;
      const modelLabel = formatAiModelLabel(modelVersion);
      setActiveModelLabel(modelLabel);
      setProgressPercent(100);
      setProgressMessage(
        `CheXNet hoàn tất — ${modelLabel} đã dự đoán 15 nhãn bệnh lý`,
      );

      message.success(`AI phân tích xong (${modelLabel})`);
      setTimeout(() => {
        onClose();
        navigate(`/patient-profile/${newPatient.patientCode}`);
      }, 800);
    } catch (err: any) {
      clearInterval(progressInterval);
      setProgressMessage('Quá trình phân tích thất bại.');
      const errMsg = err?.data?.message || err?.data || 'Lỗi không xác định khi quét hình ảnh.';
      message.error(`Quét ảnh thất bại: ${errMsg}`);
      setStep(2); // Send back to upload stage to try again
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-md p-4 transition-all duration-300">
      <div className="bg-white dark:bg-[#232736] rounded-2xl border border-outline-variant/30 dark:border-gray-800 shadow-2xl w-full max-w-2xl flex flex-col overflow-hidden transition-all transform scale-100 max-h-[90vh]">
        
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-outline-variant/20 dark:border-gray-800 shrink-0 bg-slate-50/50 dark:bg-[#1a1d27]/40 backdrop-blur">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary/10 dark:bg-blue-500/10 flex items-center justify-center">
              <span className="material-symbols-outlined text-primary dark:text-[#2563eb] text-[24px]">
                {step === 1 ? 'person_add' : step === 2 ? 'upload_file' : 'neurology'}
              </span>
            </div>
            <div>
              <h3 className="text-lg font-bold text-on-surface dark:text-white m-0">
                {step === 1 && 'Thêm Bệnh Nhân Mới'}
                {step === 2 && 'Tải X-quang ngực & Chẩn đoán phổi'}
                {step === 3 && 'AI Diagnostic Scanner'}
              </h3>
              <p className="text-xs text-on-surface-variant dark:text-gray-400 m-0 mt-0.5">
                {step === 1 && 'Đăng ký thông tin hành chính & lâm sàng của bệnh nhân'}
                {step === 2 && `Bệnh nhân: ${newPatient?.name} (${newPatient?.patientCode})`}
                {step === 3 && 'AI đang phân tích X-quang ngực (15 bệnh lý phổi)'}
              </p>
            </div>
          </div>
          
          {/* Close button (disabled in Step 3) */}
          <button 
            onClick={onClose} 
            disabled={step === 3}
            className="p-1.5 rounded-full hover:bg-slate-100 dark:hover:bg-gray-800 transition-colors disabled:opacity-30"
          >
            <span className="material-symbols-outlined text-[20px] text-on-surface-variant dark:text-gray-400">close</span>
          </button>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-6">
          
          {/* STEP 1: PATIENT INFORMATION FORM */}
          {step === 1 && (
            <form onSubmit={handleSavePatient} className="flex flex-col gap-6">
              
              {/* Section 1: Personal Info */}
              <div>
                <h4 className="text-xs font-bold uppercase tracking-wider text-primary dark:text-blue-400 mb-3 flex items-center gap-1.5">
                  <span className="material-symbols-outlined text-[16px]">badge</span>
                  Thông tin cá nhân cơ bản
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Name */}
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-semibold text-on-surface dark:text-gray-300">
                      Họ và tên bệnh nhân <span className="text-error">*</span>
                    </label>
                    <input
                      type="text"
                      required
                      placeholder="VD: Trần Thị Bích Ngọc"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      className="px-3.5 py-2.5 rounded-lg border border-outline-variant/60 dark:border-gray-700 bg-transparent text-sm focus:outline-none focus:ring-2 focus:ring-primary/40 dark:focus:ring-blue-500/40 text-on-surface dark:text-white"
                    />
                  </div>

                  {/* Age & Gender */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="flex flex-col gap-1.5">
                      <label className="text-xs font-semibold text-on-surface dark:text-gray-300">
                        Tuổi <span className="text-error">*</span>
                      </label>
                      <input
                        type="number"
                        required
                        min="0"
                        max="150"
                        placeholder="VD: 34"
                        value={age}
                        onChange={(e) => setAge(e.target.value)}
                        className="px-3.5 py-2.5 rounded-lg border border-outline-variant/60 dark:border-gray-700 bg-transparent text-sm focus:outline-none focus:ring-2 focus:ring-primary/40 dark:focus:ring-blue-500/40 text-on-surface dark:text-white"
                      />
                    </div>
                    <div className="flex flex-col gap-1.5">
                      <label className="text-xs font-semibold text-on-surface dark:text-gray-300">
                        Giới tính <span className="text-error">*</span>
                      </label>
                      <select
                        value={gender}
                        onChange={(e) => setGender(e.target.value as any)}
                        className="px-3.5 py-2.5 rounded-lg border border-outline-variant/60 dark:border-gray-700 bg-white dark:bg-[#1a1d27] text-sm focus:outline-none focus:ring-2 focus:ring-primary/40 dark:focus:ring-blue-500/40 text-on-surface dark:text-white"
                      >
                        <option value="Nam">Nam</option>
                        <option value="Nữ">Nữ</option>
                        <option value="Khác">Khác</option>
                      </select>
                    </div>
                  </div>

                  {/* Phone */}
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-semibold text-on-surface dark:text-gray-300">Số điện thoại liên lạc</label>
                    <input
                      type="tel"
                      placeholder="VD: 0987654321"
                      value={phone}
                      onChange={(e) => setPhone(e.target.value)}
                      className="px-3.5 py-2.5 rounded-lg border border-outline-variant/60 dark:border-gray-700 bg-transparent text-sm focus:outline-none focus:ring-2 focus:ring-primary/40 dark:focus:ring-blue-500/40 text-on-surface dark:text-white"
                    />
                  </div>

                  {/* Address */}
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-semibold text-on-surface dark:text-gray-300">Địa chỉ cư trú</label>
                    <input
                      type="text"
                      placeholder="VD: Quận 1, TP. Hồ Chí Minh"
                      value={address}
                      onChange={(e) => setAddress(e.target.value)}
                      className="px-3.5 py-2.5 rounded-lg border border-outline-variant/60 dark:border-gray-700 bg-transparent text-sm focus:outline-none focus:ring-2 focus:ring-primary/40 dark:focus:ring-blue-500/40 text-on-surface dark:text-white"
                    />
                  </div>
                </div>
              </div>

              {/* Section 2: Clinical Details */}
              <div>
                <h4 className="text-xs font-bold uppercase tracking-wider text-primary dark:text-blue-400 mb-3 flex items-center gap-1.5">
                  <span className="material-symbols-outlined text-[16px]">stethoscope</span>
                  Thông tin hành chính lâm sàng
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Department */}
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-semibold text-on-surface dark:text-gray-300">Khoa chẩn đoán</label>
                    <input
                      type="text"
                      placeholder="VD: Khoa Cấp cứu"
                      value={department}
                      onChange={(e) => setDepartment(e.target.value)}
                      className="px-3.5 py-2.5 rounded-lg border border-outline-variant/60 dark:border-gray-700 bg-transparent text-sm focus:outline-none focus:ring-2 focus:ring-primary/40 dark:focus:ring-blue-500/40 text-on-surface dark:text-white"
                    />
                  </div>

                  {/* Blood Type */}
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs font-semibold text-on-surface dark:text-gray-300">Nhóm máu</label>
                    <select
                      value={bloodType}
                      onChange={(e) => setBloodType(e.target.value)}
                      className="px-3.5 py-2.5 rounded-lg border border-outline-variant/60 dark:border-gray-700 bg-white dark:bg-[#1a1d27] text-sm focus:outline-none focus:ring-2 focus:ring-primary/40 dark:focus:ring-blue-500/40 text-on-surface dark:text-white"
                    >
                      <option value="A">A</option>
                      <option value="B">B</option>
                      <option value="AB">AB</option>
                      <option value="O">O</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Section 3: Expandable Vitals */}
              <div className="border border-outline-variant/40 dark:border-gray-800 rounded-xl overflow-hidden transition-all duration-300">
                <button
                  type="button"
                  onClick={() => setShowVitals(!showVitals)}
                  className="w-full flex items-center justify-between px-4 py-3 bg-slate-50 dark:bg-gray-800/40 text-left transition-colors hover:bg-slate-100 dark:hover:bg-gray-800/60"
                >
                  <span className="text-xs font-bold text-on-surface dark:text-gray-200 flex items-center gap-2">
                    <span className="material-symbols-outlined text-[18px] text-red-500">heart_check</span>
                    Nhập thông số sinh hiệu lâm sàng (Vitals - Không bắt buộc)
                  </span>
                  <span className={`material-symbols-outlined text-outline transition-transform duration-200 ${showVitals ? 'rotate-180' : ''}`}>
                    keyboard_arrow_down
                  </span>
                </button>

                {showVitals && (
                  <div className="p-4 grid grid-cols-2 sm:grid-cols-3 gap-3 bg-white dark:bg-[#232736] border-t border-outline-variant/20 dark:border-gray-800">
                    <div className="flex flex-col gap-1">
                      <label className="text-[11px] text-on-surface-variant dark:text-gray-400 font-semibold">Nhịp tim (bpm)</label>
                      <input
                        type="number"
                        placeholder="75"
                        value={heartRate}
                        onChange={(e) => setHeartRate(e.target.value)}
                        className="px-2.5 py-1.5 rounded border border-outline-variant/60 dark:border-gray-700 bg-transparent text-xs text-on-surface dark:text-white focus:outline-none focus:ring-1 focus:ring-primary"
                      />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-[11px] text-on-surface-variant dark:text-gray-400 font-semibold">Huyết áp (mmHg)</label>
                      <input
                        type="text"
                        placeholder="120/80"
                        value={bloodPressure}
                        onChange={(e) => setBloodPressure(e.target.value)}
                        className="px-2.5 py-1.5 rounded border border-outline-variant/60 dark:border-gray-700 bg-transparent text-xs text-on-surface dark:text-white focus:outline-none focus:ring-1 focus:ring-primary"
                      />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-[11px] text-on-surface-variant dark:text-gray-400 font-semibold">SpO2 (%)</label>
                      <input
                        type="number"
                        placeholder="98"
                        min="0"
                        max="100"
                        value={spo2}
                        onChange={(e) => setSpo2(e.target.value)}
                        className="px-2.5 py-1.5 rounded border border-outline-variant/60 dark:border-gray-700 bg-transparent text-xs text-on-surface dark:text-white focus:outline-none focus:ring-1 focus:ring-primary"
                      />
                    </div>
                    <div className="flex flex-col gap-1">
                      <label className="text-[11px] text-on-surface-variant dark:text-gray-400 font-semibold">Nhiệt độ (°C)</label>
                      <input
                        type="number"
                        step="0.1"
                        placeholder="36.8"
                        value={temperature}
                        onChange={(e) => setTemperature(e.target.value)}
                        className="px-2.5 py-1.5 rounded border border-outline-variant/60 dark:border-gray-700 bg-transparent text-xs text-on-surface dark:text-white focus:outline-none focus:ring-1 focus:ring-primary"
                      />
                    </div>
                    <div className="flex flex-col gap-1 col-span-2 sm:col-span-1">
                      <label className="text-[11px] text-on-surface-variant dark:text-gray-400 font-semibold">Chỉ số phổi (Lung Index)</label>
                      <input
                        type="number"
                        placeholder="90"
                        value={lungIndex}
                        onChange={(e) => setLungIndex(e.target.value)}
                        className="px-2.5 py-1.5 rounded border border-outline-variant/60 dark:border-gray-700 bg-transparent text-xs text-on-surface dark:text-white focus:outline-none focus:ring-1 focus:ring-primary"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Form Actions */}
              <div className="border-t border-outline-variant/20 dark:border-gray-800 pt-4 mt-2 flex justify-end gap-3 shrink-0">
                <button
                  type="button"
                  onClick={onClose}
                  className="px-5 py-2.5 rounded-lg border border-outline-variant bg-transparent text-sm font-semibold hover:bg-slate-50 dark:hover:bg-gray-800 text-on-surface dark:text-gray-200 transition-colors"
                >
                  Hủy bỏ
                </button>
                <button
                  type="submit"
                  disabled={isCreating}
                  className="px-5 py-2.5 rounded-lg bg-primary text-white text-sm font-semibold hover:opacity-90 transition-all flex items-center gap-1.5 shadow-md shadow-primary/10 disabled:opacity-50"
                >
                  {isCreating ? 'Đang khởi tạo...' : 'Lưu hồ sơ & Tiếp tục'}
                  <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
                </button>
              </div>
            </form>
          )}

          {/* STEP 2: SCAN FILE UPLOAD */}
          {step === 2 && (
            <div className="flex flex-col gap-6">
              
              {/* Scan Type Selection */}
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-primary dark:text-blue-400 mb-2.5 flex items-center gap-1">
                  <span className="material-symbols-outlined text-[16px]">settings_accessibility</span>
                  Tư thế chụp X-quang ngực
                </label>
                <div className="flex flex-wrap gap-2">
                  <p className="text-[11px] text-on-surface-variant dark:text-gray-400 mb-2 m-0">
                    {CHEST_XRAY_SCOPE_NOTE}
                  </p>
                  {CHEST_XRAY_SCAN_TYPES.map((type) => (
                    <button
                      key={type}
                      onClick={() => setScanType(type)}
                      className={`px-4 py-2 rounded-lg text-xs font-semibold border transition-all ${
                        scanType === type
                          ? 'bg-primary text-white border-primary shadow-sm'
                          : 'bg-background dark:bg-[#1a1d27] text-on-surface-variant dark:text-gray-400 border-outline-variant/60 dark:border-gray-700 hover:border-primary/50'
                      }`}
                    >
                      {type}
                    </button>
                  ))}
                </div>
              </div>

              {/* Drag and Drop Zone */}
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-primary dark:text-blue-400 mb-2.5 flex items-center gap-1">
                  <span className="material-symbols-outlined text-[16px]">cloud_upload</span>
                  Tải lên phim X-quang ngực (PNG/JPG/DICOM)
                </label>
                
                <div
                  onDragOver={onDragOver}
                  onDragLeave={onDragLeave}
                  onDrop={onDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`relative border-2 border-dashed rounded-2xl flex flex-col items-center justify-center gap-4 transition-all cursor-pointer min-h-[220px] p-6 ${
                    isDragging
                      ? 'border-primary bg-primary/5 dark:bg-blue-500/10'
                      : 'border-outline-variant/60 dark:border-gray-700 hover:border-primary/50 hover:bg-slate-50 dark:hover:bg-[#1a1d27]/50'
                  }`}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/jpeg,image/png,image/jpg,.dcm"
                    className="hidden"
                    onChange={onFileInputChange}
                  />

                  {previewUrl ? (
                    <div className="w-full flex flex-col items-center gap-3">
                      <img
                        src={previewUrl}
                        alt="DICOM preview"
                        className="max-h-[150px] rounded-lg object-contain border border-outline-variant/30 bg-black/5 dark:bg-black/20 p-1"
                      />
                      <div className="text-center">
                        <p className="text-xs font-bold text-primary dark:text-blue-400 truncate max-w-[300px] px-2 mx-auto">
                          {selectedFile?.name}
                        </p>
                        <p className="text-[10px] text-on-surface-variant dark:text-gray-400 mt-0.5">
                          {(selectedFile!.size / (1024 * 1024)).toFixed(2)} MB · Nhấp để thay đổi ảnh khác
                        </p>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="w-14 h-14 rounded-2xl bg-primary/10 dark:bg-blue-500/10 flex items-center justify-center">
                        <span className="material-symbols-outlined text-primary dark:text-[#2563eb] text-[36px]">
                          add_photo_alternate
                        </span>
                      </div>
                      <div className="text-center">
                        <p className="text-sm font-semibold text-on-surface dark:text-white">
                          Kéo thả phim chụp X-quang vào đây
                        </p>
                        <p className="text-xs text-on-surface-variant dark:text-gray-400 mt-1">
                          Hoặc bấm vào để duyệt file từ máy tính
                        </p>
                        <p className="text-[10px] text-outline dark:text-gray-500 mt-2 bg-slate-100 dark:bg-[#1a1d27] px-2 py-0.5 rounded inline-block">
                          Định dạng hỗ trợ: PNG, JPG, JPEG, DICOM (tối đa 10MB)
                        </p>
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Step Actions */}
              <div className="border-t border-outline-variant/20 dark:border-gray-800 pt-4 mt-2 flex justify-between shrink-0">
                <span className="text-xs text-on-surface-variant dark:text-gray-400 flex items-center gap-1 italic">
                  <span className="material-symbols-outlined text-[16px] text-emerald-500">check_circle</span>
                  Hồ sơ bệnh nhân đã được lưu trữ an toàn
                </span>
                
                <div className="flex gap-2">
                  <button
                    onClick={() => {
                      onClose();
                      navigate(`/patient-profile/${newPatient.patientCode}`);
                    }}
                    className="px-4 py-2.5 rounded-lg border border-outline-variant bg-transparent text-xs font-semibold hover:bg-slate-50 dark:hover:bg-gray-800 text-on-surface dark:text-gray-200 transition-colors"
                  >
                    Xem hồ sơ trống
                  </button>
                  <button
                    onClick={handleStartAnalysis}
                    disabled={!selectedFile || isUploading}
                    className="px-5 py-2.5 rounded-lg bg-primary text-white text-xs font-semibold hover:opacity-90 transition-all flex items-center gap-1.5 shadow-md shadow-primary/10 disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    Bắt đầu phân tích AI
                    <span className="material-symbols-outlined text-[16px] animate-pulse">neurology</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* STEP 3: PROGRESSIVE AI SCANNER */}
          {step === 3 && (
            <div className="flex flex-col items-center justify-center py-10 px-4 text-center">
              
              {/* Massive Holographic Circle Scanner */}
              <div className="relative w-40 h-40 mb-8 flex items-center justify-center">
                {/* Circular pulse waves */}
                <div className="absolute inset-0 rounded-full border border-primary/20 dark:border-blue-500/20 animate-ping duration-1000"></div>
                <div className="absolute inset-2 rounded-full border border-primary/40 dark:border-blue-500/40 animate-pulse"></div>
                
                {/* Real-time scanning glowing line */}
                <div className="absolute top-0 left-0 right-0 bottom-0 rounded-full border-4 border-slate-100 dark:border-gray-800"></div>
                <div className="absolute top-0 left-0 right-0 bottom-0 rounded-full border-t-4 border-l-4 border-primary dark:border-[#2563eb] animate-spin duration-700"></div>
                
                {/* Icon in Center */}
                <div className="w-28 h-28 rounded-full bg-slate-50 dark:bg-[#1a1d27] border border-outline-variant/30 dark:border-gray-800 flex items-center justify-center shadow-inner relative overflow-hidden">
                  <span className="material-symbols-outlined text-[54px] text-primary dark:text-[#2563eb] animate-pulse">
                    pulmonology
                  </span>
                  
                  {/* Neon sliding scanning grid overlay */}
                  <div className="absolute left-0 w-full h-1 bg-primary/80 dark:bg-[#2563eb]/80 top-0 animate-bounce shadow-[0_0_8px_rgba(37,99,235,0.8)]"></div>
                </div>
              </div>

              {/* Scanner Status Messages */}
              <div className="w-full max-w-sm flex flex-col gap-3">
                <h4 className="text-base font-bold text-on-surface dark:text-white">
                  Đang tiến hành chẩn đoán đa tiêu chí...
                </h4>
                
                {/* Horizontal Progress Bar */}
                <div className="w-full h-2.5 bg-slate-100 dark:bg-gray-800 rounded-full overflow-hidden border border-outline-variant/10">
                  <div 
                    className="h-full bg-primary dark:bg-[#2563eb] transition-all duration-300 rounded-full shadow-lg"
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
                
                <div className="flex justify-between items-center text-[11px] text-on-surface-variant dark:text-gray-400 font-mono mt-1 px-1">
                  <span className="flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-ping"></span>
                    Mô hình: {activeModelLabel}
                  </span>
                  <span className="font-bold text-primary dark:text-blue-400">{progressPercent}%</span>
                </div>

                <p className="text-sm font-semibold text-secondary dark:text-gray-300 mt-2 bg-slate-50 dark:bg-gray-850 px-4 py-2.5 rounded-xl border border-outline-variant/30 dark:border-gray-800/40 animate-pulse">
                  {progressMessage}
                </p>
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
};
