import React, { useState, useEffect } from 'react'

const SEVERITY_COLORS = {
  'Nhẹ': { bg: 'rgba(0, 229, 160, 0.15)', border: 'rgba(0, 229, 160, 0.4)', text: '#00e5a0' },
  'Trung bình': { bg: 'rgba(255, 176, 32, 0.15)', border: 'rgba(255, 176, 32, 0.4)', text: '#ffb020' },
  'Nặng': { bg: 'rgba(255, 69, 96, 0.15)', border: 'rgba(255, 69, 96, 0.4)', text: '#ff4560' },
}

export default function ClinicalReport({ report, coOccurrences, onSaveReference }) {
  const [patientId, setPatientId] = useState('')
  const [selectedDiseases, setSelectedDiseases] = useState([])
  const [isSaving, setIsSaving] = useState(false)
  const [saveMessage, setSaveMessage] = useState(null)

  if (!report) return null

  const { summary, findings, recommendations, warnings, co_occurrence_notes } = report

  useEffect(() => {
    if (findings) {
      setSelectedDiseases(findings.map(f => f.disease))
    }
  }, [findings])

  const handleDiseaseToggle = (disease) => {
    setSelectedDiseases(prev => 
      prev.includes(disease) ? prev.filter(d => d !== disease) : [...prev, disease]
    )
  }

  const handleSave = async () => {
    if (!onSaveReference) return
    setIsSaving(true)
    setSaveMessage(null)
    try {
      await onSaveReference(patientId || 'Local_Case', selectedDiseases)
      setSaveMessage({ type: 'success', text: 'Đã lưu ca bệnh vào cơ sở dữ liệu tham chiếu.' })
    } catch (err) {
      setSaveMessage({ type: 'error', text: err.message || 'Lỗi khi lưu ca bệnh.' })
    }
    setIsSaving(false)
  }

  return (
    <div className="space-y-5 animate-float-up">

      {/* Summary */}
      <div className="p-5 rounded-2xl glass-bright border border-cyan/20">
        <div className="flex items-center gap-2 mb-3">
          <span className="w-2 h-2 rounded-full bg-cyan animate-pulse" />
          <span className="text-xs font-bold text-cyan tracking-wider font-mono">
            TÓM TẮT CHẨN ĐOÁN
          </span>
        </div>
        <p className="text-sm text-text leading-relaxed">{summary}</p>
      </div>

      {/* Warnings */}
      {warnings && warnings.length > 0 && (
        <div className="space-y-2">
          {warnings.map((w, i) => (
            <div
              key={i}
              className="p-3 rounded-xl text-xs flex items-start gap-2"
              style={{
                background: 'rgba(255, 69, 96, 0.08)',
                border: '1px solid rgba(255, 69, 96, 0.3)',
                color: '#ff7b8a'
              }}
            >
              <span className="shrink-0 mt-0.5">⚠️</span>
              <span>{w}</span>
            </div>
          ))}
        </div>
      )}

      {/* Findings */}
      {findings && findings.length > 0 && (
        <div className="space-y-3">
          <div className="text-xs font-bold text-muted tracking-wider font-mono px-1">
            CHI TIẾT TỔN THƯƠNG PHÁT HIỆN
          </div>
          {findings.map((f, i) => {
            const sevColor = SEVERITY_COLORS[f.severity_vi] || SEVERITY_COLORS['Nhẹ']
            return (
              <div
                key={i}
                className="p-4 rounded-xl glass border border-cyan/15"
              >
                {/* Header */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-bold text-text">{f.disease_vi}</span>
                    <span className="text-[10px] text-muted font-mono">({f.disease})</span>
                  </div>
                  <span
                    className="px-2 py-0.5 rounded-full text-[10px] font-bold"
                    style={{ background: sevColor.bg, border: `1px solid ${sevColor.border}`, color: sevColor.text }}
                  >
                    {f.severity_vi}
                  </span>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-3 gap-2 mb-3">
                  <div className="p-2 rounded-lg bg-surface/60 border border-cyan/10 text-center">
                    <div className="text-[10px] text-muted font-mono">XÁC SUẤT</div>
                    <div className="text-sm font-bold text-cyan">{(f.probability * 100).toFixed(1)}%</div>
                    <div className="text-[9px] text-muted">{f.uncertainty_note}</div>
                  </div>
                  <div className="p-2 rounded-lg bg-surface/60 border border-cyan/10 text-center">
                    <div className="text-[10px] text-muted font-mono">TIN CẬY</div>
                    <div className={`text-sm font-bold ${
                      f.confidence_level === 'Cao' ? 'text-green' :
                      f.confidence_level === 'Trung bình' ? 'text-yellow' : 'text-red'
                    }`}>
                      {f.confidence_level}
                    </div>
                  </div>
                  <div className="p-2 rounded-lg bg-surface/60 border border-cyan/10 text-center">
                    <div className="text-[10px] text-muted font-mono">DIỆN TÍCH</div>
                    <div className="text-sm font-bold text-text">{f.area_percent?.toFixed(1) || '0.0'}%</div>
                    <div className="text-[9px] text-muted">phế trường</div>
                  </div>
                </div>

                {/* Finding text */}
                <p className="text-xs text-text/80 leading-relaxed mb-2">
                  📋 {f.finding_text}
                </p>

                {/* Location */}
                {f.location && f.location !== 'Không xác định' && (
                  <div className="text-[11px] text-muted">
                    📍 Vị trí: <span className="text-text font-medium">{f.location}</span>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}

      {/* Co-occurrence Notes */}
      {co_occurrence_notes && co_occurrence_notes.length > 0 && (
        <div className="p-4 rounded-xl glass border border-cyan/15">
          <div className="text-xs font-bold text-cyan tracking-wider font-mono mb-3">
            🔗 TƯƠNG QUAN BỆNH LÝ
          </div>
          <div className="space-y-2">
            {co_occurrence_notes.map((note, i) => (
              <div key={i} className="flex items-start gap-2 text-xs text-text/80">
                <span className="text-cyan shrink-0">›</span>
                <span>{note}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      {recommendations && recommendations.length > 0 && (
        <div className="p-4 rounded-xl glass border border-green/20" style={{ background: 'rgba(0, 229, 160, 0.05)' }}>
          <div className="text-xs font-bold tracking-wider font-mono mb-3" style={{ color: '#00e5a0' }}>
            💊 KHUYẾN NGHỊ THEO DÕI
          </div>
          <div className="space-y-2">
            {recommendations.map((rec, i) => (
              <div key={i} className="flex items-start gap-2 text-xs text-text/80">
                <span style={{ color: '#00e5a0' }} className="shrink-0">✓</span>
                <span>{rec}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Dynamic Growth (Save Reference Case) */}
      <div className="p-4 rounded-xl glass border border-cyan/30 mt-6 bg-cyan/5">
        <div className="text-sm font-bold text-cyan mb-2 flex items-center gap-2">
          <span>💾</span> Xác nhận & Lưu trữ (Dynamic Growth)
        </div>
        <p className="text-xs text-text/70 mb-4">
          Lưu ca bệnh này vào cơ sở dữ liệu FAISS nội bộ để làm tiêu chuẩn tham chiếu cho các ca bệnh sau này.
        </p>
        
        <div className="space-y-3">
          <div>
            <label className="block text-xs text-text/80 mb-1">Mã Bệnh Nhân / Hồ sơ</label>
            <input 
              type="text" 
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              placeholder="VD: BN-2023-0192"
              className="w-full bg-dark/50 border border-border rounded-lg px-3 py-2 text-sm text-text focus:outline-none focus:border-cyan"
            />
          </div>
          
          <div>
            <label className="block text-xs text-text/80 mb-2">Bệnh lý xác nhận (Ground-truth)</label>
            <div className="flex flex-wrap gap-2">
              {findings && findings.map((f, i) => (
                <button
                  key={i}
                  onClick={() => handleDiseaseToggle(f.disease)}
                  className={`px-3 py-1 text-xs rounded-full border transition-all ${
                    selectedDiseases.includes(f.disease) 
                      ? 'bg-cyan/20 border-cyan text-cyan' 
                      : 'bg-dark/50 border-border text-text/50'
                  }`}
                >
                  {f.disease_vi}
                </button>
              ))}
            </div>
          </div>

          {saveMessage && (
            <div className={`text-xs p-2 rounded ${saveMessage.type === 'success' ? 'text-green bg-green/10' : 'text-red bg-red/10'}`}>
              {saveMessage.text}
            </div>
          )}

          <button 
            onClick={handleSave}
            disabled={isSaving}
            className="w-full mt-2 bg-cyan/20 hover:bg-cyan/30 text-cyan border border-cyan/50 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
          >
            {isSaving ? 'Đang lưu...' : 'Lưu vào cơ sở dữ liệu tham chiếu'}
          </button>
        </div>
      </div>

      {/* Disclaimer */}
      <div
        className="p-3 rounded-xl text-[10px] text-center"
        style={{ background: 'rgba(255, 176, 32, 0.05)', border: '1px solid rgba(255, 176, 32, 0.15)', color: '#6a7a8a' }}
      >
        ⚕️ Báo cáo được sinh tự động bởi CheXNet AI. Mọi chẩn đoán cuối cùng cần được xác nhận bởi Bác sĩ chuyên khoa.
      </div>
    </div>
  )
}
