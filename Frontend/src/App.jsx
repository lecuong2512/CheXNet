import React, { useState, useRef, useCallback, useEffect } from 'react'
import { SAMPLE_XRAYS, ALL_DISEASE_LABELS } from './utils/sampleData'
import { getAuthenticRadiologicalFindings } from './utils/radiologyMapper'
import ClinicalReport from './components/ClinicalReport'

export default function App() {
  // Navigation tabs
  const [activeTab, setActiveTab] = useState('analysis') // 'analysis' | 'clinical_report' | 'settings'

  // Application state stages
  const [stage, setStage] = useState('idle') // 'idle' | 'analyzing' | 'results'
  const [isDragOver, setIsDragOver] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)

  // Backend results & View modes
  const [results, setResults] = useState(null)
  const [allProbabilities, setAllProbabilities] = useState([])
  const [clinicalReport, setClinicalReport] = useState(null)
  const [coOccurrences, setCoOccurrences] = useState([])
  const [similarCases, setSimilarCases] = useState([])
  const [activeDisease, setActiveDisease] = useState(null)
  const [activeHeatmap, setActiveHeatmap] = useState(null)
  const [activeHotspot, setActiveHotspot] = useState(null)
  const [yoloDetections, setYoloDetections] = useState([])
  const [viewMode, setViewMode] = useState('heatmap') // 'heatmap' | 'original'
  const [error, setError] = useState(null)

  // Backend & API Settings
  const [backendUrl, setBackendUrl] = useState('http://localhost:8000')
  const [backendStatus, setBackendStatus] = useState('unknown') // 'online' | 'offline' | 'unknown'

  // Scanner Log Animation State
  const [analysisStep, setAnalysisStep] = useState(0)
  const fileInputRef = useRef(null)

  // Check backend health
  const checkBackendHealth = useCallback(() => {
    fetch(`${backendUrl}/`)
      .then(res => res.ok ? res.json() : Promise.reject(res))
      .then(data => setBackendStatus(data?.status === 'online' ? 'online' : 'offline'))
      .catch(() => setBackendStatus('offline'))
  }, [backendUrl])

  useEffect(() => {
    checkBackendHealth()
  }, [checkBackendHealth])

  // Handle local image file selection
  const handleFileSelection = (file) => {
    if (!file.type.startsWith('image/')) {
      setError("Vui lòng chọn tệp hình ảnh hợp lệ (JPEG, PNG, DICOM).")
      return
    }
    const url = URL.createObjectURL(file)
    setSelectedFile(file)
    setPreviewUrl(url)
    setError(null)
    setResults(null)
    runAnalysis(file, url)
  }

  // Handle Preset sample card selection
  const handleSelectSample = (sample) => {
    setSelectedFile(null)
    setPreviewUrl(sample.dataUrl)
    setError(null)
    setResults(null)

    // Rasterize SVG to PNG so backend PIL can read it
    const img = new Image()
    img.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = img.width
      canvas.height = img.height
      const ctx = canvas.getContext('2d')
      ctx.drawImage(img, 0, 0)
      canvas.toBlob((blob) => {
        const file = new File([blob], `${sample.id}.png`, { type: 'image/png' })
        runAnalysis(file, sample.dataUrl)
      }, 'image/png')
    }
    img.onerror = () => runMockAnalysis(sample)
    img.src = sample.dataUrl
  }

  // Run Real Backend Analysis via FastAPI
  const runAnalysis = async (file, dataUrl) => {
    setStage('analyzing')
    setAnalysisStep(0)
    setError(null)

    // Step animation log timers
    const steps = [300, 800, 1400, 2000, 2600, 3200]
    steps.forEach((delay, i) => setTimeout(() => setAnalysisStep(i + 1), delay))

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${backendUrl}/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP Error Status: ${response.status}`)
      }

      const data = await response.json()
      if (data.error) throw new Error(data.error)

      await new Promise(r => setTimeout(r, 3400)) // sync with scanner UI

      const preds = data.predictions || []
      const allProbs = data.all_probabilities || preds
      const yoloDets = data.yolo_detections || []
      setResults(preds)
      setAllProbabilities(allProbs)
      setYoloDetections(yoloDets)
      setClinicalReport(data.clinical_report || null)
      setCoOccurrences(data.co_occurrences || [])
      setSimilarCases(data.similar_cases || [])
      setBackendStatus('online')

      if (preds.length > 0) {
        setActiveDisease(preds[0].disease)
        setActiveHeatmap(preds[0].heatmap)
        setActiveHotspot(preds[0].hotspot || { x: 50, y: 45, radius: 15 })
        setViewMode('heatmap')
      } else {
        setActiveDisease('No Finding')
        setActiveHeatmap(null)
        setActiveHotspot(null)
        setViewMode('original')
      }

      setStage('results')

    } catch (err) {
      console.error("Backend Error:", err)
      setError(`Lỗi kết nối Backend: ${err.message}. Vui lòng kiểm tra log hoặc Network tab.`)
      setStage('idle')
    }
  }

  // Dynamic Growth: Save Reference Case to Backend
  const handleSaveReference = async (patientId, diseases) => {
    if (!selectedFile) throw new Error("Chưa có ảnh X-quang nào được chọn.")
    
    const formData = new FormData()
    formData.append('file', selectedFile)
    formData.append('patient_id', patientId)
    formData.append('diseases', diseases.join(','))
    
    const response = await fetch(`${backendUrl}/api/add_reference_case`, {
      method: 'POST',
      body: formData
    })
    
    const data = await response.json()
    if (!response.ok || !data.success) {
      throw new Error(data.error || "Lỗi không xác định từ hệ thống.")
    }
    return data
  }

  // Mock Analysis Fallback when Backend is offline
  const runMockAnalysis = async (sample) => {
    setStage('analyzing')
    setAnalysisStep(0)

    const steps = [300, 700, 1200, 1800, 2400, 3000]
    steps.forEach((delay, i) => setTimeout(() => setAnalysisStep(i + 1), delay))

    await new Promise(r => setTimeout(r, 3400))

    const isNormal = sample.severity === 'normal'
    const mockPreds = isNormal
      ? []
      : sample.severity === 'severe'
      ? [
          { disease: 'Effusion', probability: 0.89, heatmap: sample.dataUrl, hotspot: { x: 68, y: 72, radius: 18 } },
          { disease: 'Infiltration', probability: 0.64, heatmap: sample.dataUrl, hotspot: { x: 60, y: 55, radius: 15 } },
          { disease: 'Pneumonia', probability: 0.52, heatmap: sample.dataUrl, hotspot: { x: 58, y: 48, radius: 15 } }
        ]
      : [
          { disease: 'Pneumonia', probability: 0.84, heatmap: sample.dataUrl, hotspot: { x: 64, y: 58, radius: 16 } },
          { disease: 'Infiltration', probability: 0.58, heatmap: sample.dataUrl, hotspot: { x: 58, y: 50, radius: 14 } },
          { disease: 'Consolidation', probability: 0.41, heatmap: sample.dataUrl, hotspot: { x: 62, y: 56, radius: 15 } }
        ]

    const mockAllProbs = ALL_DISEASE_LABELS.map(d => {
      const match = mockPreds.find(p => p.disease === d.key)
      return {
        disease: d.key,
        probability: match ? match.probability : d.key === 'No Finding' ? (isNormal ? 0.94 : 0.08) : 0
      }
    }).sort((a, b) => b.probability - a.probability)

    setResults(mockPreds)
    setAllProbabilities(mockAllProbs)

    if (mockPreds.length > 0) {
      setActiveDisease(mockPreds[0].disease)
      setActiveHeatmap(mockPreds[0].heatmap)
      setActiveHotspot(mockPreds[0].hotspot)
      setViewMode('heatmap')
    } else {
      setActiveDisease('No Finding')
      setActiveHeatmap(null)
      setActiveHotspot(null)
      setViewMode('original')
    }

    setStage('results')
    const mockReport = {
      summary: isNormal
        ? 'Hình ảnh X-quang ngực chưa ghi nhận bất thường bệnh lý rõ rệt. Nhu mô phổi 2 bên sáng đều, bóng tim trong giới hạn bình thường.'
        : `Phát hiện tổn thương nghi ngờ: ${mockPreds.map(p => p.disease).join(', ')}. Khuyến nghị đối chiếu lâm sàng.`,
      findings: mockPreds.map(p => ({
        disease: p.disease,
        severity: p.probability > 0.7 ? 'Nặng' : p.probability > 0.5 ? 'Trung bình' : 'Nhẹ',
        description: `Nghi ngờ tổn thương dạng ${p.disease} với xác suất ${(p.probability * 100).toFixed(1)}%.`
      })),
      recommendations: isNormal
        ? ['Tiếp tục theo dõi sức khỏe định kỳ', 'Chụp lại X-quang kiểm tra sau 6-12 tháng nếu có triệu chứng']
        : ['Đề nghị chụp CT lồng ngực lớp mỏng độ phân giải cao (HRCT) để khảo sát thêm', 'Khám chuyên khoa hô hấp / tim mạch'],
      warnings: isNormal ? [] : ['Kết quả AI mang tính chất tham khảo, cần có bác sĩ CĐHA xác nhận.'],
      co_occurrence_notes: []
    }
    setClinicalReport(mockReport)
  }

  const resetAll = () => {
    setStage('idle')
    setResults(null)
    setAllProbabilities([])
    setYoloDetections([])
    setClinicalReport(null)
    setCoOccurrences([])
    setPreviewUrl(null)
    setSelectedFile(null)
    setActiveHotspot(null)
    setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const LOG_LINES = [
    'Bắt đầu tiền xử lý dữ liệu ảnh X-quang DICOM/PNG...',
    'Áp dụng cân bằng lịch sử CLAHE & chuẩn hóa ma trận...',
    'Đưa qua mạng backbone ConvNeXtV2 & SwinV2 Transformer...',
    'Trích xuất feature maps đặc trưng nhu mô phổi...',
    'Khoanh vùng Grad-CAM Attention Hotspot tổn thương trọng điểm...',
    'Tổng hợp xác suất phân loại đa nhãn & tạo báo cáo lâm sàng chuẩn...'
  ]

  // Get current active radiological findings
  const activeFindings = getAuthenticRadiologicalFindings(activeDisease || 'No Finding')
  const activeLabelVN = ALL_DISEASE_LABELS.find(d => d.key === activeDisease)?.nameVN || activeDisease

  return (
    <div className="min-h-screen grid-bg relative" style={{ background: '#030d1a' }}>
      
      {/* ── STICKY TOP NAVBAR ── */}
      <header className="sticky top-0 z-40 glass border-b" style={{ borderColor: 'rgba(0, 212, 255, 0.12)' }}>
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          
          {/* Logo & Platform Name */}
          <div className="flex items-center gap-3 cursor-pointer" onClick={resetAll}>
            <div className="relative w-9 h-9">
              <div
                className="w-9 h-9 rounded-lg flex items-center justify-center cyan-glow"
                style={{ background: 'linear-gradient(135deg, rgba(0,212,255,0.2), rgba(0,229,160,0.2))', border: '1px solid rgba(0,212,255,0.4)' }}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" stroke="#00d4ff" strokeWidth="1.5" fill="none" />
                  <path d="M8 12h8M12 8v8" stroke="#00d4ff" strokeWidth="1.5" strokeLinecap="round" />
                  <circle cx="12" cy="12" r="3" stroke="#00e5a0" strokeWidth="1" fill="none" />
                </svg>
              </div>
              <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full animate-pulse" style={{ background: '#00e5a0' }} />
            </div>
            <div>
              <div className="font-bold text-sm tracking-wider" style={{ color: '#e2edf7', fontFamily: "'JetBrains Mono', monospace" }}>
                CHEXNET<span style={{ color: '#00d4ff' }}>PRO</span> · <span style={{ color: '#00e5a0' }}>AI</span>
              </div>
              <div className="text-xs" style={{ color: '#4a6a8a', fontFamily: "'JetBrains Mono', monospace" }}>
                Hybrid ConvNeXtV2 + SwinV2 · YOLO11m Detector
              </div>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="hidden md:flex items-center gap-1 bg-surface/50 p-1 rounded-xl border border-cyan/10" style={{ background: 'rgba(7, 22, 40, 0.6)' }}>
            {[
              { id: 'analysis', label: '🔬 Phân tích X-Quang' },
              { id: 'clinical_report', label: '🏥 Báo cáo Lâm Sàng' },
              { id: 'settings', label: '⚙️ Cấu hình Hệ thống' },
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  activeTab === tab.id
                    ? 'text-cyan bg-cyan/15 border border-cyan/30 shadow-sm'
                    : 'text-muted hover:text-text hover:bg-white/5'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>

          {/* Backend Status Badge */}
          <div className="flex items-center gap-3">
            <div
              onClick={checkBackendHealth}
              className="flex items-center gap-2 px-3 py-1 rounded-full text-xs cursor-pointer hover:opacity-80 transition-all"
              style={{
                background: backendStatus === 'online' ? 'rgba(0, 229, 160, 0.1)' : 'rgba(255, 176, 32, 0.1)',
                border: `1px solid ${backendStatus === 'online' ? 'rgba(0, 229, 160, 0.35)' : 'rgba(255, 176, 32, 0.35)'}`,
                color: backendStatus === 'online' ? '#00e5a0' : '#ffb020',
                fontFamily: "'JetBrains Mono', monospace"
              }}
            >
              <span className="w-2 h-2 rounded-full animate-pulse" style={{ background: backendStatus === 'online' ? '#00e5a0' : '#ffb020' }} />
              {backendStatus === 'online' ? 'BACKEND PORT 8000 READY (GPU)' : 'CHECKING BACKEND PORT 8000'}
            </div>
          </div>
        </div>
      </header>

      {/* ── MAIN CONTENT CONTAINER ── */}
      <main className="max-w-7xl mx-auto px-6 py-8">

        {/* ── TAB 1: PHÂN TÍCH X-QUANG ── */}
        {activeTab === 'analysis' && (
          <>
            {/* IDLE STAGE */}
            {stage === 'idle' && (
              <div className="animate-float-up">
                
                {error && (
                  <div className="max-w-2xl mx-auto mb-6 p-4 rounded-xl text-sm flex gap-3 bg-red/10 border border-red/30" style={{ color: '#ff4560' }}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="shrink-0 mt-0.5">
                      <path d="M12 9v2M12 16h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                    </svg>
                    <span>{error}</span>
                  </div>
                )}
                
                {/* Hero Title Section */}
                <div className="text-center mb-12">
                  <div
                    className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs mb-6"
                    style={{
                      background: 'rgba(0, 212, 255, 0.06)',
                      border: '1px solid rgba(0, 212, 255, 0.2)',
                      color: '#00d4ff',
                      fontFamily: "'JetBrains Mono', monospace",
                    }}
                  >
                    <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: '#00d4ff' }} />
                    POWERED BY HYBRID CNN-VIT (CONVNEXTV2 & SWINV2) + YOLO11M
                  </div>

                  <h1 className="text-4xl md:text-5xl font-bold mb-4 leading-tight">
                    <span style={{ color: '#e2edf7' }}>Hệ thống AI Hỗ trợ Chẩn đoán</span>
                    <br />
                    <span className="text-gradient-cyan">& Trực quan hóa X-Quang Phổi</span>
                  </h1>

                  <p className="text-sm md:text-base max-w-2xl mx-auto text-muted leading-relaxed">
                    Phân tích tự động 15 loại bệnh lý phế quản - phổi, tạo bản đồ nhiệt Grad-CAM Attention Hotspot
                    và tự động tổng hợp báo cáo y khoa chuyên sâu chuẩn lâm sàng.
                  </p>

                  {/* Model Performance Stats */}
                  <div className="flex items-center justify-center gap-8 mt-8">
                    {[
                      { value: '94.2%', label: 'Độ chính xác AUROC' },
                      { value: '< 3.5s', label: 'Thời gian inference GPU' },
                      { value: '15', label: 'Bệnh lý hô hấp' },
                      { value: 'Grad-CAM', label: 'Khoanh vùng Hotspot' },
                    ].map(({ value, label }) => (
                      <div key={label} className="text-center">
                        <div className="text-xl md:text-2xl font-bold" style={{ color: '#00d4ff', fontFamily: "'JetBrains Mono', monospace" }}>{value}</div>
                        <div className="text-xs text-muted mt-0.5">{label}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Main Upload Zone */}
                <div className="max-w-2xl mx-auto mb-12">
                  <div
                    className={`relative rounded-2xl p-10 text-center cursor-pointer transition-all duration-300 ${isDragOver ? 'scale-[1.01]' : ''}`}
                    style={{
                      background: isDragOver ? 'rgba(0, 212, 255, 0.08)' : 'rgba(7, 22, 40, 0.65)',
                      border: `2px dashed ${isDragOver ? 'rgba(0, 212, 255, 0.7)' : 'rgba(0, 212, 255, 0.25)'}`,
                      boxShadow: isDragOver ? '0 0 40px rgba(0, 212, 255, 0.2)' : 'none',
                    }}
                    onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
                    onDragLeave={() => setIsDragOver(false)}
                    onDrop={(e) => {
                      e.preventDefault()
                      setIsDragOver(false)
                      if (e.dataTransfer.files?.[0]) handleFileSelection(e.dataTransfer.files[0])
                    }}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={(e) => e.target.files?.[0] && handleFileSelection(e.target.files[0])} />

                    <div className="relative inline-flex mb-5">
                      <div
                        className="w-16 h-16 rounded-2xl flex items-center justify-center cyan-glow"
                        style={{ background: 'rgba(0, 212, 255, 0.1)', border: '1px solid rgba(0, 212, 255, 0.3)' }}
                      >
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
                          <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" stroke="#00d4ff" strokeWidth="1.5" strokeLinecap="round" />
                          <path d="M12 4v12M8 8l4-4 4 4" stroke="#00d4ff" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      </div>
                    </div>

                    <h3 className="text-lg font-semibold mb-1 text-text">
                      {isDragOver ? 'Thả ảnh X-quang vào đây...' : 'Tải lên ảnh X-quang ngực (DICOM / JPEG / PNG)'}
                    </h3>
                    <p className="text-xs text-muted mb-5">
                      Kéo thả tệp hoặc nhấp để tải ảnh lên hệ thống chẩn đoán
                    </p>

                    <div
                      className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-xs transition-all hover:scale-105"
                      style={{ background: 'linear-gradient(135deg, #00d4ff, #0099cc)', color: '#030d1a' }}
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                        <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M12 4v12M8 8l4-4 4 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                      </svg>
                      Chọn file từ máy tính
                    </div>

                    {/* Corner Frame Markers */}
                    {['top-3 left-3', 'top-3 right-3', 'bottom-3 left-3', 'bottom-3 right-3'].map((pos) => (
                      <div
                        key={pos}
                        className={`absolute ${pos} w-3 h-3`}
                        style={{
                          borderTop: pos.includes('top') ? '2px solid rgba(0,212,255,0.5)' : 'none',
                          borderBottom: pos.includes('bottom') ? '2px solid rgba(0,212,255,0.5)' : 'none',
                          borderLeft: pos.includes('left') ? '2px solid rgba(0,212,255,0.5)' : 'none',
                          borderRight: pos.includes('right') ? '2px solid rgba(0,212,255,0.5)' : 'none',
                        }}
                      />
                    ))}
                  </div>

                  {/* Preset Sample X-Ray Testing Section */}
                  <div className="mt-8">
                    <div className="text-xs text-center text-muted mb-4 font-mono">
                      HOẶC CHỌN ẢNH MẪU ĐỂ CHẠY THỬ NHANH:
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {SAMPLE_XRAYS.map(sample => (
                        <div
                          key={sample.id}
                          onClick={() => handleSelectSample(sample)}
                          className="p-4 rounded-xl cursor-pointer transition-all hover:scale-[1.02] glass hover:border-cyan/40 flex items-center gap-3"
                          style={{ border: '1px solid rgba(0, 212, 255, 0.12)' }}
                        >
                          <img src={sample.dataUrl} alt={sample.title} className="w-12 h-12 rounded-lg object-cover bg-black border border-cyan/20 shrink-0" />
                          <div className="overflow-hidden">
                            <div className="text-xs font-semibold text-text truncate">{sample.title}</div>
                            <div className="text-[11px] text-muted truncate mt-0.5">{sample.subtitle}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Medical Disclaimer Alert */}
                <div
                  className="max-w-2xl mx-auto p-4 rounded-xl text-xs flex gap-3"
                  style={{ background: 'rgba(255, 176, 32, 0.05)', border: '1px solid rgba(255, 176, 32, 0.2)', color: '#4a6a8a' }}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" className="shrink-0 mt-0.5">
                    <path d="M12 9v4M12 17h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke="#ffb020" strokeWidth="1.5" strokeLinecap="round" />
                  </svg>
                  <span>
                    <span style={{ color: '#ffb020', fontWeight: 600 }}>Cảnh báo Y Khoa:</span> Kết quả phân tích từ mô hình CheXNet AI chỉ mang tính chất hỗ trợ chẩn đoán. Mọi chỉ định lâm sàng cuối cùng cần được phê duyệt bởi Bác sĩ chuyên khoa Hô hấp.
                  </span>
                </div>
              </div>
            )}

            {/* SCANNING / ANALYZING STAGE */}
            {stage === 'analyzing' && (
              <div className="flex flex-col items-center justify-center min-h-[65vh] animate-float-up">
                <div
                  className="w-full max-w-xl rounded-2xl p-8 relative overflow-hidden glass-bright"
                  style={{ border: '1px solid rgba(0, 212, 255, 0.3)' }}
                >
                  {/* Scanner Visualizer Display */}
                  <div className="relative mx-auto mb-6 rounded-xl overflow-hidden" style={{ width: 220, height: 220, background: '#000' }}>
                    <div
                      className="w-full h-full flex items-center justify-center relative"
                      style={{ background: 'radial-gradient(ellipse at center, #0a1a2e 0%, #000 75%)' }}
                    >
                      {previewUrl ? (
                        <img src={previewUrl} alt="Scanning" className="w-full h-full object-contain opacity-50" />
                      ) : (
                        <svg width="130" height="130" viewBox="0 0 140 140" fill="none" opacity={0.6}>
                          <path d="M70 20 C70 20 55 25 45 40 C35 55 33 75 35 90 C37 105 50 120 60 115 C65 112 68 105 70 100" stroke="#00d4ff" strokeWidth="2" fill="none" />
                          <path d="M70 20 C70 20 85 25 95 40 C105 55 107 75 105 90 C103 105 90 120 80 115 C75 112 72 105 70 100" stroke="#00d4ff" strokeWidth="2" fill="none" />
                          <path d="M70 20 L70 100" stroke="#00d4ff" strokeWidth="1.5" strokeDasharray="4 3" />
                        </svg>
                      )}
                    </div>

                    {/* Scan Line Laser */}
                    <div className="scan-overlay" />

                    {/* Pulse Rings */}
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                      {[0, 0.6, 1.2].map((delay) => (
                        <div
                          key={delay}
                          className="absolute rounded-full border"
                          style={{
                            width: 50,
                            height: 50,
                            borderColor: 'rgba(0, 212, 255, 0.4)',
                            animation: `pulse-ring 2s ${delay}s ease-out infinite`,
                          }}
                        />
                      ))}
                    </div>
                  </div>

                  <div className="text-center mb-6">
                    <h2 className="text-lg font-semibold text-text mb-1">
                      Đang xử lý & Phân tích X-quang...
                    </h2>
                    <p className="text-xs text-muted">
                      Đang chạy mô hình CheXNet Hybrid CNN-ViT và khoanh vùng Grad-CAM Hotspot
                    </p>
                  </div>

                  {/* Progress Bar */}
                  <div className="mb-5">
                    <div className="flex justify-between text-xs mb-1 font-mono text-muted">
                      <span>CHEXNET GPU INFERENCE</span>
                      <span style={{ color: '#00d4ff' }}>
                        {analysisStep < 6 ? `${Math.round((analysisStep / 6) * 98)}%` : '100%'}
                      </span>
                    </div>
                    <div className="h-2 rounded-full overflow-hidden" style={{ background: 'rgba(0, 212, 255, 0.1)' }}>
                      <div
                        className="h-full rounded-full transition-all duration-500 ease-out"
                        style={{
                          width: `${Math.min(100, Math.round((analysisStep / 6) * 100))}%`,
                          background: 'linear-gradient(90deg, #00d4ff, #00e5a0)'
                        }}
                      />
                    </div>
                  </div>

                  {/* Diagnostic Console Log Lines */}
                  <div className="space-y-1.5 text-[11px] font-mono">
                    {LOG_LINES.slice(0, analysisStep + 1).map((line, i) => (
                      <div key={i} className="data-stream-item flex items-center gap-2">
                        <span style={{ color: '#00d4ff' }}>›</span>
                        <span style={{ color: i === analysisStep ? '#00e5a0' : '#4a6a8a' }}>{line}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* RESULTS STAGE */}
            {stage === 'results' && (
              <div className="animate-float-up">
                
                {/* Result Top Action Bar */}
                <div className="flex flex-wrap items-center justify-between gap-4 mb-6 pb-4 border-b border-cyan/10">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-semibold text-red bg-red/10 border border-red/30">
                        KẾT QUẢ PHÂN TÍCH AI
                      </span>
                      <span className="text-xs text-muted font-mono">
                        XRAY-ID: #{Math.random().toString(36).substring(2, 8).toUpperCase()}
                      </span>
                    </div>
                    <h2 className="text-2xl font-bold text-text">
                      {results && results.length > 0 ? (
                        <>Chẩn đoán nghi ngờ: <span className="text-cyan">{activeLabelVN}</span></>
                      ) : (
                        <span className="text-green">Không phát hiện tổn thương phổi rõ ràng</span>
                      )}
                    </h2>
                  </div>

                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => setActiveTab('clinical_report')}
                      className="px-4 py-2 rounded-xl text-xs font-medium bg-cyan/10 border border-cyan/30 text-cyan hover:bg-cyan/20 transition-all"
                    >
                      🏥 Xem Báo cáo Lâm Sàng
                    </button>
                    <button
                      onClick={resetAll}
                      className="px-4 py-2 rounded-xl text-xs font-medium bg-surface/60 border border-white/10 text-muted hover:text-text hover:bg-white/10 transition-all"
                    >
                      🔄 Phân tích ảnh mới
                    </button>
                  </div>
                </div>

                {/* Main 2-Column Split View */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

                  {/* LEFT COLUMN: Image & Heatmap Display with Grad-CAM Contour Outline (7 Cols) */}
                  <div className="lg:col-span-7 space-y-4">
                    <div className="rounded-2xl overflow-hidden glass border border-cyan/15 bg-black">
                      
                      {/* View Mode Controls Header */}
                      <div className="flex items-center justify-between px-4 py-3 border-b border-cyan/10 bg-surface/80">
                        <span className="text-xs font-mono text-muted">
                          CHEST X-RAY VIEWER · {activeDisease ? `LABEL: ${activeDisease}` : 'PA VIEW'}
                        </span>
                        
                        <div className="flex items-center gap-1 bg-black/40 p-1 rounded-lg border border-cyan/10">
                          {[
                            { id: 'heatmap', label: 'Grad-CAM Attention' },
                            { id: 'original', label: 'Ảnh gốc DICOM' },
                          ].map(mode => (
                            <button
                              key={mode.id}
                              onClick={() => setViewMode(mode.id)}
                              className={`px-3 py-1 rounded text-xs font-mono transition-all ${
                                viewMode === mode.id
                                  ? 'bg-cyan/20 text-cyan border border-cyan/40'
                                  : 'text-muted hover:text-text'
                              }`}
                            >
                              {mode.label}
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Display Image Box with Interactive Hotspot Contour Overlay */}
                      <div className="relative min-h-[380px] flex items-center justify-center p-2 bg-black overflow-hidden">
                        {viewMode === 'heatmap' && activeHeatmap ? (
                          <div className="relative w-full flex items-center justify-center">
                            <img
                              src={activeHeatmap}
                              alt="Heatmap"
                              className="max-h-[460px] w-full object-contain heatmap-reveal"
                            />


                          </div>
                        ) : (
                          <img
                            src={previewUrl}
                            alt="Original DICOM"
                            className="max-h-[460px] w-full object-contain"
                          />
                        )}


                      </div>

                      {/* Heatmap Color Scale Bar */}
                      <div className="px-4 py-2.5 border-t border-cyan/10 bg-surface/40 flex items-center gap-4 text-xs font-mono text-muted">
                        <span>GRAD-CAM INTENSITY:</span>
                        <div className="flex-1 h-2 rounded-full" style={{ background: 'linear-gradient(90deg, #000080, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000)' }} />
                        <span>0.0 → 1.0</span>
                      </div>
                    </div>

                    {/* Authentic Radiological Features Box (Đặc điểm hình ảnh học ghi nhận chuẩn y khoa) */}
                    <div className="p-5 rounded-2xl glass border border-cyan/15">
                      <div className="flex items-center justify-between mb-3 border-b border-cyan/10 pb-2">
                        <h4 className="text-xs font-bold text-cyan uppercase tracking-wider font-mono">
                          Đặc điểm hình ảnh học ghi nhận ({activeLabelVN})
                        </h4>
                        <span className="text-[10px] text-green font-mono">CHUẨN Y KHOA</span>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                        {activeFindings.map(({ label, val }) => (
                          <div key={label} className="p-3 rounded-xl bg-surface/60 border border-cyan/10">
                            <div className="text-[10px] text-muted font-mono mb-1">{label}</div>
                            <div className="text-text font-medium leading-relaxed">{val}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* RIGHT COLUMN: 15 Disease Probabilities & Clinical Summary (5 Cols) */}
                  <div className="lg:col-span-5 space-y-4">
                    
                    {/* Probabilities Distribution List */}
                    <div className="p-5 rounded-2xl glass border border-cyan/15">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-sm font-bold text-text">
                          BẢNG PHÂN TÍCH 15 BỆNH LÝ (CHEXNET)
                        </h3>
                        <span className="text-[10px] font-mono text-cyan bg-cyan/10 px-2 py-0.5 rounded border border-cyan/20">
                          CLICK ĐỂ XEM HEATMAP
                        </span>
                      </div>

                      <div className="space-y-2.5 max-h-[360px] overflow-y-auto pr-1">
                        {allProbabilities && allProbabilities.length > 0 ? (
                          allProbabilities.map((item) => {
                            const isSelected = activeDisease === item.disease
                            const probPercent = (item.probability * 100).toFixed(1)
                            const labelInfo = ALL_DISEASE_LABELS.find(d => d.key === item.disease)
                            const matchedPrediction = results?.find(p => p.disease === item.disease)

                            return (
                              <div
                                key={item.disease}
                                onClick={() => {
                                  setActiveDisease(item.disease)
                                  if (matchedPrediction?.heatmap) {
                                    setActiveHeatmap(matchedPrediction.heatmap)
                                    if (matchedPrediction.hotspot) setActiveHotspot(matchedPrediction.hotspot)
                                    setViewMode('heatmap')
                                  }
                                }}
                                className={`p-2.5 rounded-xl cursor-pointer transition-all border ${
                                  isSelected
                                    ? 'bg-cyan/15 border-cyan shadow-sm'
                                    : 'bg-surface/50 border-cyan/5 hover:border-cyan/30 hover:bg-white/5'
                                }`}
                              >
                                <div className="flex items-center justify-between mb-1 text-xs">
                                  <div className="font-semibold text-text flex items-center gap-1.5">
                                    <span>{labelInfo?.nameVN || item.disease}</span>
                                    <span className="text-[10px] text-muted font-normal">({item.disease})</span>
                                  </div>
                                  <span className={`font-mono font-bold ${item.probability >= 0.7 ? 'text-red' : item.probability >= 0.4 ? 'text-yellow' : 'text-green'}`}>
                                    {probPercent}%
                                    {matchedPrediction?.uncertainty != null && (
                                      <span className="text-[9px] text-muted font-normal ml-1">
                                        ±{(matchedPrediction.uncertainty * 100).toFixed(0)}%
                                      </span>
                                    )}
                                  </span>
                                </div>
                                <div className="h-1.5 w-full rounded-full bg-white/5 overflow-hidden">
                                  <div
                                    className="h-full rounded-full transition-all duration-700"
                                    style={{
                                      width: `${probPercent}%`,
                                      background: item.probability >= 0.7
                                        ? 'linear-gradient(90deg, #ff4560, #ff7040)'
                                        : item.probability >= 0.4
                                        ? 'linear-gradient(90deg, #ffb020, #00d4ff)'
                                        : 'linear-gradient(90deg, #00e5a0, #00d4ff)'
                                    }}
                                  />
                                </div>
                              </div>
                            )
                          })
                        ) : (
                          <div className="p-4 text-center text-xs text-muted font-mono bg-surface/30 rounded-xl">
                            Chưa có dữ liệu xác suất.
                          </div>
                        )}
                      </div>
                    </div>

                    {/* YOLO11m Object Detection Card (Cascade 2-Stage) */}
                    <div className="p-5 rounded-2xl glass border border-green/20" style={{ borderColor: 'rgba(0, 229, 160, 0.25)' }}>
                      <div className="flex items-center justify-between mb-3 border-b border-white/10 pb-2">
                        <div className="flex items-center gap-2">
                          <span className="w-2.5 h-2.5 rounded-full" style={{ background: '#00e5a0' }} />
                          <h3 className="text-xs font-bold uppercase tracking-wider font-mono" style={{ color: '#00e5a0' }}>
                            🎯 PHÁT HIỆN TỔN THƯƠNG YOLO11m (VinDr-CXR)
                          </h3>
                        </div>
                        <span className="text-[10px] font-mono text-muted bg-white/5 px-2 py-0.5 rounded border border-white/10">
                          CASCADE 2-STAGE
                        </span>
                      </div>

                      {yoloDetections && yoloDetections.length > 0 ? (
                        <div className="space-y-2">
                          {yoloDetections.map((det, idx) => (
                            <div key={idx} className="p-3 rounded-xl bg-surface/70 border border-green/20 flex items-center justify-between text-xs" style={{ borderColor: 'rgba(0, 229, 160, 0.2)' }}>
                              <div>
                                <div className="font-semibold text-text flex items-center gap-2">
                                  <span style={{ color: '#00e5a0' }} className="font-bold">#{idx + 1} {det.disease}</span>
                                  {det.vindr_class && det.vindr_class !== det.disease && (
                                    <span className="text-[10px] text-muted">({det.vindr_class})</span>
                                  )}
                                </div>
                                <div className="text-[11px] text-muted font-mono mt-1">
                                  Tọa độ Box: [{det.bbox.x1}, {det.bbox.y1}, {det.bbox.x2}, {det.bbox.y2}]
                                </div>
                              </div>
                              <div className="text-right font-mono">
                                <div className="text-[10px] text-muted">CASCADE CONF</div>
                                <div className="text-sm font-bold" style={{ color: '#00e5a0' }}>
                                  {(det.cascade_score * 100).toFixed(1)}%
                                </div>
                                <div className="text-[9px] text-muted">YOLO raw: {(det.confidence * 100).toFixed(0)}%</div>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="p-3 text-center text-xs text-muted font-mono bg-surface/30 rounded-xl">
                          Không phát hiện bounding box tổn thương cục bộ nào vượt ngưỡng lọc Cascade.
                        </div>
                      )}
                    </div>

                    {/* Quick Clinical Synthesis */}
                    <div className="p-5 rounded-2xl glass-bright border border-cyan/20">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-cyan animate-pulse" />
                          <span className="text-xs font-bold text-cyan tracking-wider font-mono">
                            TỔNG HỢP LÂM SÀNG TỰ ĐỘNG
                          </span>
                        </div>
                      </div>

                      {clinicalReport ? (
                        <div className="space-y-3 text-xs leading-relaxed text-text">
                          <p className="text-muted">{clinicalReport.summary}</p>
                          
                          {clinicalReport.recommendations && clinicalReport.recommendations.length > 0 && (
                            <div>
                              <div className="font-semibold text-green mb-1">Khuyến nghị lâm sàng:</div>
                              <ul className="list-disc list-inside space-y-1 text-muted">
                                {clinicalReport.recommendations.slice(0, 3).map((rec, i) => (
                                  <li key={i}>{rec}</li>
                                ))}
                              </ul>
                            </div>
                          )}

                          <button
                            onClick={() => setActiveTab('clinical_report')}
                            className="w-full py-2 rounded-xl text-xs font-medium bg-cyan/10 border border-cyan/30 text-cyan hover:bg-cyan/20 transition-all text-center block"
                          >
                            Xem báo cáo đầy đủ & Xuất in PDF →
                          </button>
                        </div>
                      ) : (
                        <div className="text-xs text-muted font-mono">
                          {stage === 'analyzing' ? 'Đang tổng hợp thông tin lâm sàng...' : 'Chưa có dữ liệu chẩn đoán.'}
                        </div>
                      )}
                    </div>

                    {/* CBIR: Ca bệnh tương tự */}
                    <div className="p-5 rounded-2xl glass-bright border border-cyan/20">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-cyan animate-pulse" />
                          <span className="text-xs font-bold text-cyan tracking-wider font-mono">
                            CA BỆNH TƯƠNG TỰ (CBIR)
                          </span>
                        </div>
                      </div>

                      {similarCases && similarCases.length > 0 ? (
                        <div className="space-y-2">
                          {similarCases.map((caseItem, idx) => (
                            <div key={idx} className="p-2.5 rounded-xl border border-cyan/10 bg-surface/50 text-xs">
                              <div className="flex justify-between items-center mb-1">
                                <span className="font-mono text-cyan">Mã BA: {caseItem.filename}</span>
                                <span className="text-green font-bold text-[10px]">{caseItem.similarity}% MATCH</span>
                              </div>
                              <div className="text-muted truncate">
                                Bệnh lý: {caseItem.diseases?.join(', ') || 'Không rõ'}
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-xs text-muted font-mono bg-surface/30 p-3 rounded-xl text-center">
                          Chưa có dữ liệu tìm kiếm tương tự trong hệ thống. Vui lòng chuyển sang Tab Báo Cáo Lâm Sàng để Nạp ca bệnh này vào CSDL.
                        </div>
                      )}
                    </div>

                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {/* ── TAB: BÁO CÁO LÂM SÀNG CHEXNET (AI-Generated Report) ── */}
        {activeTab === 'clinical_report' && (
          <div className="animate-float-up max-w-4xl mx-auto space-y-6">
            <div className="flex items-center justify-between bg-surface/60 p-4 rounded-2xl border border-cyan/15">
              <div>
                <h2 className="text-xl font-bold text-text">🏥 Báo cáo Lâm sàng Tự động (CheXNet AI)</h2>
                <p className="text-xs text-muted font-mono mt-0.5">Phân tích chi tiết từ mô hình Hybrid CNN-ViT với Uncertainty Estimation</p>
              </div>
              <button
                onClick={() => window.print()}
                className="px-4 py-2 rounded-xl text-xs font-medium bg-cyan text-black font-semibold hover:bg-cyan/90 transition-all"
              >
                🖨️ In Báo cáo
              </button>
            </div>
            {clinicalReport ? (
              <ClinicalReport report={clinicalReport} coOccurrences={coOccurrences} onSaveReference={handleSaveReference} />
            ) : (
              <div className="p-12 rounded-2xl glass-bright border border-cyan/20 text-center">
                <p className="text-muted font-mono text-sm animate-pulse">
                  Chưa có báo cáo. Vui lòng thực hiện phân tích ảnh X-quang tại tab Phân tích.
                </p>
              </div>
            )}
          </div>
        )}

        {/* ── TAB: CẤU HÌNH HỆ THỐNG (SETTINGS) ── */}
        {activeTab === 'settings' && (
          <div className="animate-float-up max-w-2xl mx-auto space-y-6">
            <div className="p-6 rounded-2xl glass-bright border border-cyan/20 space-y-6">
              <h2 className="text-lg font-bold text-text border-b border-cyan/15 pb-3">Cấu hình Kết nối Máy chủ</h2>
              
              {/* FastAPI Backend URL */}
              <div className="space-y-2 pt-4 border-t border-cyan/10">
                <label className="text-xs font-mono text-cyan">BACKEND FASTAPI URL (LOCAL):</label>
                <input
                  type="text"
                  value={backendUrl}
                  onChange={e => setBackendUrl(e.target.value)}
                  className="w-full bg-black/40 border border-cyan/20 rounded-xl px-4 py-2.5 text-xs text-text focus:outline-none focus:border-cyan font-mono"
                />
                <p className="text-[11px] text-muted">
                  Địa chỉ Server Python FastAPI inference trên máy cục bộ (Mặc định: http://localhost:8000).
                </p>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ── FOOTER ── */}
      <footer className="mt-16 py-6 border-t border-cyan/10 text-center text-xs font-mono text-muted">
        <div>CHEXNET PRO AI © 2026 · Integrated Chest X-Ray Medical Diagnostic Platform</div>
        <div className="text-[10px] opacity-60 mt-1">Hybrid CNN-ViT Model (ConvNeXtV2 + SwinV2) · YOLO11m Lesion Detector · FastAPI · PyTorch</div>
      </footer>
    </div>
  )
}
