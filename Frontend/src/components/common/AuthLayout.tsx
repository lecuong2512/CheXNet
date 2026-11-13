import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';

export default function AuthLayout({ children }: { children: ReactNode }) {
  return <main className="auth-page">
    <aside className="auth-intro" aria-label="Giới thiệu hệ thống">
      <div className="auth-wordmark">CheXNet<span>Hệ thống X-quang ngực</span></div>
      <div className="auth-intro-copy">
        <p className="auth-eyebrow">HỖ TRỢ CHẨN ĐOÁN HÌNH ẢNH</p>
        <h2>Mỗi lần chụp,<br /><span>một hồ sơ rõ ràng.</span></h2>
        <p>Đọc ảnh X-quang, đối chiếu kết quả phân tích và theo dõi hồ sơ bệnh nhân trong cùng một nơi.</p>
        <figure className="auth-illustration" aria-label="Minh họa phim X-quang ngực">
          <div className="auth-paper" aria-hidden="true">
            <span>PHIẾU KẾT QUẢ</span>
            <i /><i /><i />
            <div className="auth-paper-rule" />
            <i /><i />
          </div>
          <div className="auth-film">
            <div className="auth-film-toolbar"><span>X-quang ngực</span><span>PA</span></div>
            <svg viewBox="0 0 300 255" fill="none" aria-hidden="true">
              <defs>
                <linearGradient id="auth-lung-fill" x1="150" y1="30" x2="150" y2="230" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#a8c3da" stopOpacity=".24" /><stop offset="1" stopColor="#a8c3da" stopOpacity=".04" />
                </linearGradient>
              </defs>
              <path d="M144 45C121 42 103 50 87 71C70 95 61 141 65 182C67 201 75 218 91 220C105 222 124 211 135 201C143 193 142 174 141 154Z" fill="url(#auth-lung-fill)" stroke="#9cb8cd" strokeWidth="1.6" />
              <path d="M156 45C179 42 197 50 213 71C230 95 239 141 235 182C233 201 225 218 209 220C195 222 176 211 165 201C157 193 158 174 159 154Z" fill="url(#auth-lung-fill)" stroke="#9cb8cd" strokeWidth="1.6" />
              <path d="M150 21V107M150 88L124 117M150 88L176 117" stroke="#c6d8e5" strokeWidth="5" strokeLinecap="round" opacity=".65" />
              <path d="M150 114V232" stroke="#93acc1" strokeWidth="10" strokeLinecap="round" opacity=".13" />
              {[0, 1, 2, 3, 4, 5].map((rib) => <g key={rib} transform={`translate(0 ${rib * 23})`} stroke="#bdd0df" strokeWidth="1.6" opacity={.5 - rib * .04}>
                <path d="M139 65C116 57 85 62 75 81M161 65C184 57 215 62 225 81" />
              </g>)}
              <path d="M139 45L109 32L85 41M161 45L191 32L215 41" stroke="#bdcfdd" strokeWidth="2" opacity=".55" />
              <text x="25" y="32" fill="#8ca6bb" fontSize="10" fontFamily="sans-serif">R</text>
            </svg>
            <figcaption>Ảnh minh họa</figcaption>
          </div>
        </figure>
        <ul className="auth-features">
          <li><span>01</span>Quản lý hồ sơ</li>
          <li><span>02</span>Phân tích ảnh</li>
          <li><span>03</span>Xuất kết quả</li>
        </ul>
      </div>
      <p className="auth-copyright">© {new Date().getFullYear()} CheXNet</p>
    </aside>
    <section className="auth-main" aria-label="Tài khoản">
      <div className="auth-mobile-wordmark">CheXNet</div>
      <div className="auth-form">{children}</div>
      <Link to="/huong-dan" className="auth-help">Hướng dẫn sử dụng</Link>
    </section>
  </main>;
}
