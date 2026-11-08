"""Render artifacts for evaluate_public_pneumonia.py; never changes predictions."""
from pathlib import Path
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'.runtime/external-eval-2026-11-08'


def main():
    metrics=json.loads((OUTPUT/'metrics.json').read_text(encoding='utf-8'))
    rows=[json.loads(line) for line in (OUTPUT/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    valid=[r for r in rows if r.get('classifier_success')]
    truth=[r['label'] for r in valid]
    score=[r['probabilities']['Pneumonia'] for r in valid]
    predicted=[int(p>=0.5) for p in score]
    matrix=confusion_matrix(truth,predicted,labels=[0,1])
    primary=metrics['primary_classifier_pneumonia_threshold_0_5']
    independent={'accuracy':float(accuracy_score(truth,predicted)),
                 'sensitivity':float(recall_score(truth,predicted,zero_division=0)),
                 'specificity':float(recall_score(truth,predicted,pos_label=0,zero_division=0)),
                 'precision':float(precision_score(truth,predicted,zero_division=0)),
                 'f1':float(f1_score(truth,predicted,zero_division=0)),
                 'roc_auc':float(roc_auc_score(truth,score)), 'confusion_matrix':matrix.tolist()}
    for key in ('accuracy','sensitivity','specificity','f1','roc_auc'):
        assert math.isclose(independent[key],primary[key] or 0,abs_tol=1e-12),(key,independent[key],primary[key])
    assert matrix.tolist()==[[primary['TN'],primary['FP']],[primary['FN'],primary['TP']]]
    secondary_checks={}
    for key,decision in [
        ('web_native_pneumonia_on_accepted_images',lambda r:'Pneumonia' in r['native_findings']),
        ('secondary_any_native_finding_for_abnormality',lambda r:bool(r['native_findings'])),
        ('secondary_no_finding_rule_for_abnormality',lambda r:r['probabilities']['No Finding']<.5),
    ]:
        cohort=[r for r in valid if r.get('web_success')] if key!='secondary_no_finding_rule_for_abnormality' else valid
        actual=float(accuracy_score([r['label'] for r in cohort],[int(decision(r)) for r in cohort]))
        assert math.isclose(actual,metrics[key]['accuracy'],abs_tol=1e-12)
        secondary_checks[key]=actual
    independent['secondary_accuracy_checks']=secondary_checks
    (OUTPUT/'independent-metric-check.json').write_text(json.dumps(independent,indent=2),encoding='utf-8')

    fig,axes=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1,1.2]})
    axes[0].imshow(matrix,cmap='Blues',vmin=0,vmax=max(100,matrix.max()))
    axes[0].set_xticks([0,1],['Negative','Pneumonia'])
    axes[0].set_yticks([0,1],['NORMAL','PNEUMONIA'])
    axes[0].set_xlabel('Model decision: Pneumonia score >= 0.5')
    axes[0].set_ylabel('Dataset label')
    axes[0].set_title(f'Correct: {primary["TP"]+primary["TN"]}/{primary["n"]} = {primary["accuracy"]:.1%}')
    names=[['TN','FP'],['FN','TP']]
    for y in range(2):
        for x in range(2):
            axes[0].text(x,y,f'{names[y][x]}\n{matrix[y,x]}',ha='center',va='center',fontsize=20,
                         color='white' if matrix[y,x]>50 else '#16324f')
    bins=np.linspace(0,1,21)
    for label,color,name in [(0,'#2563eb','NORMAL (100)'),(1,'#dc2626','PNEUMONIA (100)')]:
        axes[1].hist([r['probabilities']['Pneumonia'] for r in valid if r['label']==label],bins=bins,
                     color=color,alpha=.5,label=name,edgecolor='white')
    axes[1].axvline(.5,color='#111827',linestyle='--',label='Current threshold: 0.5')
    axes[1].set_xlabel('Pneumonia score returned by model')
    axes[1].set_ylabel('Number of images')
    axes[1].set_xlim(0,1)
    axes[1].set_title(f'AUROC = {primary["roc_auc"]:.3f}')
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('External test: Kermany pediatric X-rays | 200 images | fixed seed 20261108',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.94])
    fig.savefig(OUTPUT/'evaluation-summary.png',dpi=150,bbox_inches='tight')
    fig.savefig(OUTPUT/'evaluation-summary.svg',bbox_inches='tight')
    plt.close(fig)

    pct=lambda number: 'N/A' if number is None else f'{number*100:.2f}%'
    ci=lambda bounds: 'N/A' if bounds is None else f'{pct(bounds[0])}–{pct(bounds[1])}'
    system=metrics['system']
    normals=metrics['normal_images']
    native=metrics['web_native_pneumonia_on_accepted_images']
    nf=metrics['secondary_no_finding_rule_for_abnormality']
    abnormal=metrics['secondary_any_native_finding_for_abnormality']
    checkpoint=json.loads((OUTPUT/'checkpoint.json').read_text(encoding='utf-8-sig'))
    manifest=json.loads((OUTPUT/'manifest.json').read_text(encoding='utf-8'))
    gateway=json.loads((OUTPUT/'gateway-verification.json').read_text(encoding='utf-8'))
    report=f'''# Đánh giá tạm thời Hybrid bằng 200 ảnh có nhãn

Ngày 08/11/2026. Model hiện chạy trên CPU tại `D:\\NCKH\\CheXNet`. Đây là đánh giá ngoại bộ cho **nhãn Pneumonia**, không phải accuracy tổng thể của 14 bệnh.

## Thiết kế được cố định trước khi chạy

- 100 NORMAL và 100 PNEUMONIA từ thư mục test của bộ Kermany. Không dùng thư mục train/val của nguồn để chấm điểm.
- Chọn ngẫu nhiên với seed 20261108, một ảnh trên mỗi nhóm mã bệnh nhân suy ra từ tên file. Mã nhóm chưa được kiểm chứng độc lập bằng hồ sơ bệnh nhân.
- Chọn từ 234 ảnh NORMAL / 225 nhóm tên và 390 ảnh PNEUMONIA / 202 nhóm tên trong bản sao dữ liệu đã ghim commit. 200 SHA-256 khác nhau.
- Kiểm tra tên file với Dataset/train_list.csv và val_list.csv tại workspace: không trùng. Điều này không chứng minh checkpoint chưa từng thấy ảnh, vì chưa có đầy đủ nguồn gốc dữ liệu huấn luyện của checkpoint.
- Ngưỡng chính: xác suất Pneumonia >= 0,5 là dương tính, đúng ngưỡng đang chạy. Không đổi threshold, trọng số, tiền xử lý hoặc kỹ thuật gốc dựa trên mẫu này.
- Gọi POST /v3/predict, cùng adapter/validator và pipeline AI mà gateway web đang gọi. Không tạo 200 hồ sơ bệnh nhân trong database. Lưu đủ 15 xác suất và nhãn báo cáo theo từng ảnh.
- Giữ mọi ảnh đã chọn. Nếu API v3 từ chối 422, ghi nhận lỗi và gọi /predict gốc để chấm riêng classifier; không tính ảnh bị web từ chối thành chẩn đoán đúng. Số lượt fallback thực tế: {system['legacy_fallback_count']}.

## Chỉ số chính: có nhận diện đúng lớp Pneumonia hay không

| Chỉ số | Kết quả | Khoảng tin cậy 95% Wilson |
|---|---:|---:|
| Accuracy | {pct(primary['accuracy'])} ({primary['TP']+primary['TN']}/{primary['n']}) | {ci(primary['accuracy_ci95_wilson'])} |
| Độ nhạy với viêm phổi | {pct(primary['sensitivity'])} ({primary['TP']}/{primary['TP']+primary['FN']}) | {ci(primary['sensitivity_ci95_wilson'])} |
| Độ đặc hiệu của lớp viêm phổi | {pct(primary['specificity'])} ({primary['TN']}/{primary['TN']+primary['FP']}) | {ci(primary['specificity_ci95_wilson'])} |
| Precision trên mẫu cân bằng | {pct(primary['precision'])} | — |
| F1 | {primary['f1']:.4f} | — |
| AUROC | {primary['roc_auc']:.4f} | Chưa tính |

| Nhãn nguồn / Dự đoán lớp Pneumonia | Âm tính | Dương tính |
|---|---:|---:|
| NORMAL | {primary['TN']} | {primary['FP']} |
| PNEUMONIA | {primary['FN']} | {primary['TP']} |

Âm tính ở bảng này chỉ có nghĩa **Pneumonia dưới ngưỡng**, không có nghĩa báo cáo toàn bộ model xác nhận phổi bình thường. Accuracy của bộ cân bằng này bằng balanced accuracy; precision không đại diện cho tỷ lệ mắc bệnh thực tế. Một bộ dự đoán luôn âm tính cũng đạt 50% accuracy trên mẫu này, vì vậy phải đọc cùng độ nhạy và AUROC.

Đã tính đối chiếu độc lập bằng sklearn: confusion matrix, accuracy, sensitivity, specificity, F1 và AUROC trùng phép tính báo cáo. Bằng chứng independent-metric-check.json.

## Kết luận mà giao diện thực sự hiển thị

- Nhãn Pneumonia có trong danh sách native top-six: accuracy {pct(native['accuracy'])}, sensitivity {pct(native['sensitivity'])}, specificity {pct(native['specificity'])}, trên {native['n']} ảnh được adapter chấp nhận. Hai ảnh viêm phổi có Pneumonia >= 0,5 nhưng đứng hạng 7 trong các nhãn bệnh nên bị loại khỏi kết luận tóm tắt: person1664_virus_2877.jpeg (57,87%) và person19_virus_50.jpeg (51,19%). Tab đủ 15 xác suất vẫn chứa các giá trị này. Bằng chứng pneumonia-dropped-from-top-six.json.
- Trong {normals['n']} ảnh NORMAL, pipeline vẫn báo ít nhất một bệnh trên **{normals['any_native_pathology']} ảnh**.
- Có {normals['no_finding_ge_0_5']} ảnh NORMAL có No Finding >= 0,5; trong đó {normals['contradiction_no_finding_and_pathology']} ảnh đồng thời được báo có bệnh. Đây là bất đồng giữa các đầu ra trong cùng pipeline, cần được kiểm tra riêng.
- Quy tắc phụ “có bất kỳ nhãn bệnh native = bất thường” cho accuracy **{pct(abnormal['accuracy'])} ({abnormal['TP']+abnormal['TN']}/{abnormal['n']})**, khoảng tin cậy 95% {ci(abnormal['accuracy_ci95_wilson'])}, độ nhạy {pct(abnormal['sensitivity'])}, độ đặc hiệu {pct(abnormal['specificity'])}; quy tắc phụ “No Finding < 0,5 = bất thường” cho accuracy {pct(nf['accuracy'])}. Nhận ra có bất thường không đồng nghĩa xác định đúng bệnh. Đây là các phép đọc bổ sung đã khai báo trước, không thay thế hoặc tối ưu lại tiêu chí chính.
- Bộ dữ liệu chỉ ghi NORMAL/PNEUMONIA. Không dùng các ảnh PNEUMONIA để kết luận đúng/sai cho 13 bệnh khác chưa được gán nhãn đầy đủ.

## Tính hoạt động và thời gian

- Hoàn tất {metrics['sample']['completed']}/{metrics['sample']['planned']} ảnh đã chọn.
- API v3 nhận và trả thành công {system['web_accepted']}/{metrics['sample']['planned']} ảnh; classifier chấm được {system['classifier_scored']}/{metrics['sample']['planned']} ảnh.
- Lượt lỗi hoặc bị từ chối: {len(system['failed_or_rejected'])}. Danh sách đầy đủ trong metrics.json, không loại khỏi mẫu đã chọn.
- Thời gian HTTP v3 thành công: median {system['wall_seconds_median']:.3f} giây/ảnh; p95 {system['wall_seconds_p95']:.3f} giây/ảnh. Tổng thời gian các yêu cầu (kể cả fallback nếu có): {system['wall_seconds_sum']:.1f} giây. Đây là tải tuần tự trên CPU, không phải đo tải nhiều người dùng.
- YOLO chưa có trọng số và không tham gia phép chấm này. CBIR có ca thử cũ, không dùng điểm truy hồi hoặc nhãn ca tham chiếu để tính accuracy.
- Sau khi chấm xong, gửi thêm hai ảnh đầu tiên của hai lớp qua gateway Node có đăng nhập, tạo hai scan và đọc lại từ database: cả hai HTTP 201, trạng thái done, preview HTTP 200, đủ 15 xác suất và sai khác lớn nhất so với lượt chấm bằng 0. Đây là kiểm tra tích hợp qua HTTP, không thực hiện lại thao tác giao diện trong lượt 200 ảnh.
- Hồ sơ demo được giữ để xem trên web: **{gateway['patientCode']}**, tên TEST 200 Kermany - HO SO GIA. Tuổi và giới tính trong hồ sơ là dữ liệu giả để thử phần mềm; hai ảnh không thuộc một bệnh nhân. Hai ca không được xác minh là kết luận bác sĩ hoặc thêm vào CBIR. Chi tiết gateway-verification.json.
- Kiểm tra cuối: frontend, gateway và AI đều HTTP 200. 25 tệp kỹ thuật gốc theo manifest được đối chiếu SHA-256, không tệp nào đổi. Các kiểm tra API và báo cáo không chỉnh model hoặc ứng dụng đang chạy.

## Giới hạn diễn giải

Nguồn là X-quang trẻ em 1–5 tuổi. README của dự án mô tả huấn luyện NIH/VinDr-CXR, nên có thể có khác biệt phân phối; phép thử này đo kết quả thật trên mẫu đã chọn nhưng không xác định nguyên nhân sai. Không suy rộng chỉ số này thành hiệu quả trên bệnh nhân trưởng thành, DICOM từ máy chụp hoặc mọi bệnh trong báo cáo. Khoảng tin cậy chỉ phản ánh lấy mẫu nhị thức gần đúng, không bao gồm sai nhãn và khác biệt phân phối. Không sửa model sau khi xem kết quả trong lượt này.

## Nguồn và khả năng chạy lại

- Tác giả dữ liệu: Daniel Kermany, Kang Zhang, Michael Goldbaum (2018), [Mendeley Data V2](https://data.mendeley.com/datasets/rscbjbr9sj/2), DOI 10.17632/rscbjbr9sj.2. Ảnh CC BY 4.0.
- Ảnh tải từ [probml/chest_xray_kaggle](https://github.com/probml/chest_xray_kaggle/tree/{metrics['source_commit']}/test), bản sao có dẫn nguồn Kaggle; commit {metrics['source_commit']}.
- Mô tả tuổi, nhãn và split 624 ảnh: [dataset card hf-vision](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia/blob/main/README.md).
- Checkpoint SHA-256: `{checkpoint['Hash'].lower()}`.
- Protocol SHA-256: `{metrics['protocol_sha256']}`. protocol.json được ghi trước suy luận, không thay đổi trong lượt chấm.
- Thư mục bằng chứng: `D:\\NCKH\\CheXNet\\.runtime\\external-eval-2026-11-08\\`. Có protocol, manifest kèm URL/hash từng ảnh, source tree, ảnh nguyên bản, results.jsonl, predictions.csv, metrics.json và health trước/sau.

Chạy từ workspace bằng Python trong .venv:

```powershell
.\\.venv\\Scripts\\python.exe Backend/tools/evaluate_public_pneumonia.py prepare
.\\.venv\\Scripts\\python.exe Backend/tools/evaluate_public_pneumonia.py download
.\\.venv\\Scripts\\python.exe Backend/tools/evaluate_public_pneumonia.py run
.\\.venv\\Scripts\\python.exe Backend/tools/evaluate_public_pneumonia.py report
.\\.venv\\Scripts\\python.exe Backend/tools/summarize_public_evaluation.py
```

Lệnh run đọc lại results.jsonl để tiếp tục phần chưa xong; không gọi lại các ảnh đã có kết quả. Muốn đo lại độc lập phải dùng --output với thư mục mới và lưu nguồn gốc checkpoint tương ứng.
'''
    report_path=ROOT/'docs/danh-gia-200-anh-2026-11-08.md'
    report_path.write_text(report,encoding='utf-8')
    print(json.dumps({'report':str(report_path),'independent_metrics':independent},indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
