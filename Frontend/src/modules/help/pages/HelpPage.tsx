import { Button } from 'antd';
import { useNavigate } from 'react-router-dom';

export default function HelpPage() {
  const navigate = useNavigate();
  return <div className="page-shell space-y-6 max-w-5xl">
    <header className="page-heading"><div><p className="eyebrow">BẮT ĐẦU VỚI CHEXNET</p><h1>Hướng dẫn sử dụng</h1><p className="page-description">Theo dõi một hồ sơ từ lúc tải ảnh đến khi bác sĩ xác minh kết quả.</p></div></header>
    <ol className="space-y-4">{[
      ['Thêm hồ sơ và tải ảnh', 'Chọn “Thêm bệnh nhân”, nhập thông tin bắt buộc rồi tiếp tục. Chọn tư thế chụp và tải một ảnh JPEG, PNG hoặc DICOM, tối đa 10 MB. Với bệnh nhân đã có hồ sơ, mở hồ sơ và chọn “Tải X-quang ngực”.'],
      ['Xem ảnh và kết quả', 'Ở “Phân tích X-quang”, chọn bệnh nhân và lần chụp. Ảnh gốc hiển thị trước. Chuyển sang “Bản đồ chú ý” để xem vùng mô hình tập trung; chọn dấu hiệu cần đối chiếu. Có thể phóng to, tăng tương phản và đảo màu.'],
      ['Đọc đủ điểm dự đoán', 'Mở thẻ “Điểm AI” để xem đủ 15 nhãn. Điểm của một nhãn không phải độ chính xác của hệ thống. Kết quả và bản đồ chú ý cần được đối chiếu với ảnh gốc và thông tin lâm sàng.'],
      ['Xác minh và xuất phiếu', 'Mở phiếu kết quả, nhập kết luận bác sĩ rồi chọn “Lưu và xác minh”. Lưu thay đổi trước khi tải PDF hoặc in. Chỉ tài khoản bác sĩ hoặc quản trị viên có quyền xác minh.'],
      ['Tra cứu lại và bổ sung ca tham chiếu', 'Dùng “Hồ sơ & lịch sử” để tìm tên hoặc mã bệnh nhân và mở đúng lần chụp. Sau khi xác minh phiếu, bác sĩ có thể chọn nhãn xác nhận để lưu ca tham chiếu cho chức năng tìm ảnh tương tự.'],
    ].map(([title, text], index) => <li key={title} className="ui-card flex items-start gap-4"><span className="w-8 h-8 shrink-0 flex items-center justify-center bg-blue-50 dark:bg-blue-900/30 text-primary rounded-full font-bold">{index + 1}</span><div><h2 className="section-title">{title}</h2><p className="page-description">{text}</p></div></li>)}</ol>
    <section className="ui-card space-y-3"><h2 className="section-title">Khi gặp sự cố</h2><p className="text-sm leading-relaxed">Nếu ảnh bị từ chối, kiểm tra định dạng, dung lượng và nội dung X-quang ngực. Nếu hệ thống chưa sẵn sàng, mở trang trạng thái để kiểm tra kết nối và mô hình. Với kết quả cũ, tải lại trang hoặc chọn lại lần chụp.</p>
      <div className="flex flex-wrap gap-3"><Button type="primary" onClick={() => navigate('/')}>Mở trang phân tích</Button><Button onClick={() => navigate('/system-config')}>Kiểm tra hệ thống</Button></div></section>
  </div>;
}
