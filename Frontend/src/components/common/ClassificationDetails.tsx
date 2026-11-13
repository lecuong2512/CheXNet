import type { ClassificationSummary } from '../../types/hybrid';
import { diseaseLabel, formatPercent } from '../../utils/display';

export default function ClassificationDetails({ classification, showFindings = false }: {
  classification?: ClassificationSummary; showFindings?: boolean;
}) {
  if (!classification) return <p className="text-xs leading-relaxed text-secondary" role="note">
    Phiếu cũ chưa lưu ngưỡng phân loại. Danh sách tóm tắt chỉ gồm các nhãn có bản đồ chú ý; xem “Điểm AI” để đối chiếu đủ 15 nhãn.
  </p>;
  const hideNoFindingConflict = classification.findings.length > 0;
  const visibleWarnings = hideNoFindingConflict
    ? classification.warnings.filter(warning => !warning.includes('không phát hiện bất thường'))
    : classification.warnings;
  const highlightConflict = classification.conflictingNoFinding && !hideNoFindingConflict;
  return <div className="space-y-3 text-sm">
    {showFindings && <>
      <h3 className="font-semibold">Danh sách nhãn vượt ngưỡng</h3>
      <p>{classification.summary}</p>
      <p className="text-xs text-secondary">Tên nhãn là nhóm phân loại; kích thước ghi trong tên nhãn không phải số đo từ ảnh.</p>
      <ul className="space-y-2">{classification.findings.map(item => <li key={item.disease} className="flex justify-between gap-4">
        <span>{diseaseLabel(item.disease)}</span>
        <span className="text-right shrink-0">{formatPercent(item.probability)}<span className="block text-xs text-secondary">Ngưỡng {formatPercent(item.threshold)}</span></span>
      </li>)}</ul>
    </>}
    {visibleWarnings.length > 0 && <div role="note" className={`border-l-2 pl-3 space-y-2 text-xs leading-relaxed ${highlightConflict ? 'border-amber-500 text-amber-800 dark:text-amber-300' : 'border-slate-300 text-secondary dark:text-gray-400'}`}>
      {visibleWarnings.map(warning => <p key={warning}>{warning}</p>)}
    </div>}
  </div>;
}
