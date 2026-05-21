import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { useDispatch } from "react-redux";
import { Form, Input, Button, message, Spin } from "antd";
import { useLoginMutation } from "../../../stores/baseApi";
import { setCredentials } from "../../../stores/authSlice";

const LoginPage: React.FC = () => {
  const [form] = Form.useForm();
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const [login, { isLoading }] = useLoginMutation();

  const fillDemoAccount = (email: string, password: string) => {
    form.setFieldsValue({ email, password });
  };

  const onFinish = async (values: any) => {
    try {
      const response = await login({
        email: String(values.email || "").trim().toLowerCase(),
        password: values.password,
      }).unwrap();
      if (response?.success && response?.data) {
        const { user, accessToken, refreshToken } = response.data;
        dispatch(setCredentials({ user, accessToken, refreshToken }));
        const greeting =
          user.role === "admin"
            ? `Đăng nhập thành công! Chào Quản trị viên ${user.name}`
            : `Đăng nhập thành công! Chào mừng bác sĩ ${user.name}`;
        message.success({
          content: greeting,
          duration: 3,
        });
        navigate("/");
      } else {
        message.error("Đăng nhập thất bại. Vui lòng thử lại.");
      }
    } catch (err: any) {
      const errMsg = err?.data?.message || err?.message || "Đăng nhập thất bại. Vui lòng kiểm tra lại thông tin.";
      message.error(errMsg);
    }
  };

  return (
    <div className="flex h-screen w-screen bg-[#f8fafc] dark:bg-[#151722] text-gray-900 dark:text-gray-100 overflow-hidden font-sans">
      {/* Left side: Premium hospital visual brand & AI info */}
      <div className="hidden lg:flex lg:w-7/12 relative flex-col justify-between p-12 bg-medical-grid bg-[#0c1033] dark:bg-[#0b0d1b] border-r border-slate-200/10 text-white select-none overflow-hidden">
        {/* Animated ambient glow circles */}
        <div className="absolute top-[-10%] right-[-10%] w-[500px] h-[500px] rounded-full bg-[#004ac6] opacity-25 blur-[120px] pointer-events-none" />
        <div className="absolute bottom-[-10%] left-[-10%] w-[400px] h-[400px] rounded-full bg-[#10b981] opacity-15 blur-[120px] pointer-events-none" />

        {/* Brand header */}
        <div className="flex items-center gap-3 z-10">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-[#004ac6] to-[#00d2ff] flex items-center justify-center shadow-lg shadow-blue-500/20">
            <span className="material-symbols-outlined text-white text-[24px]">clinical_notes</span>
          </div>
          <div>
            <div className="text-[20px] font-bold tracking-wider bg-gradient-to-r from-white via-[#e2e8f0] to-[#94a3b8] bg-clip-text text-transparent">CHEXNET AI</div>
            <div className="text-[10px] text-blue-400 font-medium tracking-widest uppercase">Trợ lý Phân tích X-quang Ngực</div>
          </div>
        </div>

        {/* Interactive Neural Net Node / X-Ray Chest scanning visualization */}
        <div className="my-auto flex flex-col items-center justify-center relative z-10">
          <div className="relative w-[360px] h-[360px] border border-blue-500/20 bg-slate-900/40 rounded-3xl p-6 shadow-2xl flex items-center justify-center overflow-hidden backdrop-blur-md">
            {/* Medical scanning grid overlay */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(0,74,198,0.05)_1px,transparent_1px),linear-gradient(to_bottom,rgba(0,74,198,0.05)_1px,transparent_1px)] bg-[size:16px_16px]" />
            
            {/* Animated Laser Scanning Line */}
            <div className="absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-[#00d2ff] to-transparent shadow-[0_0_15px_#00d2ff] animate-[pulse_2s_infinite] top-[10%] animate-[scan_3s_linear_infinite]" />

            {/* Premium Chest X-Ray SVG Vector illustration */}
            <svg width="240" height="240" viewBox="0 0 240 240" fill="none" className="opacity-80 drop-shadow-[0_0_10px_rgba(0,210,255,0.2)]">
              {/* Spine */}
              <rect x="117" y="20" width="6" height="200" rx="3" fill="#cbd5e1" opacity="0.3" />
              {/* Rib cage left */}
              <path d="M110,40 C80,40 50,55 45,90 C40,120 50,150 55,190 C80,200 100,195 110,190" stroke="#cbd5e1" strokeWidth="3" strokeLinecap="round" opacity="0.4" />
              <path d="M110,70 C90,70 65,80 60,110 C55,130 62,150 65,180" stroke="#cbd5e1" strokeWidth="2.5" strokeLinecap="round" opacity="0.3" />
              <path d="M110,100 C95,100 75,110 72,130 C70,145 74,160 76,175" stroke="#cbd5e1" strokeWidth="2" strokeLinecap="round" opacity="0.3" />
              
              {/* Rib cage right */}
              <path d="M130,40 C160,40 190,55 195,90 C200,120 190,150 185,190 C160,200 140,195 130,190" stroke="#cbd5e1" strokeWidth="3" strokeLinecap="round" opacity="0.4" />
              <path d="M130,70 C150,70 175,80 180,110 C185,130 178,150 175,180" stroke="#cbd5e1" strokeWidth="2.5" strokeLinecap="round" opacity="0.3" />
              <path d="M130,100 C145,100 165,110 168,130 C170,145 166,160 164,175" stroke="#cbd5e1" strokeWidth="2" strokeLinecap="round" opacity="0.3" />

              {/* Heart Shadow */}
              <path d="M117,100 C100,105 90,120 95,140 C100,160 117,175 117,175 C117,175 125,160 128,150 C130,140 125,120 117,100 Z" fill="#38bdf8" opacity="0.15" />
              
              {/* AI Detection boxes / overlays */}
              <rect x="52" y="80" width="50" height="60" rx="8" stroke="#10b981" strokeWidth="1.5" strokeDasharray="4 2" fill="rgba(16, 185, 129, 0.05)" />
              <circle cx="77" cy="110" r="4" fill="#10b981" className="animate-ping" />
              <text x="56" y="94" fill="#10b981" fontSize="8" fontWeight="bold" fontFamily="monospace">AI PASS</text>
              
              <rect x="135" y="110" width="50" height="50" rx="8" stroke="#ef4444" strokeWidth="1.5" strokeDasharray="4 2" fill="rgba(239, 68, 68, 0.05)" />
              <circle cx="160" cy="135" r="4" fill="#ef4444" className="animate-ping" />
              <text x="139" y="124" fill="#ef4444" fontSize="8" fontWeight="bold" fontFamily="monospace">MASS 89%</text>
            </svg>
          </div>
          
          <div className="mt-8 text-center max-w-sm">
            <h2 className="text-xl font-bold text-slate-100">Đột phá Y tế từ Trí tuệ Nhân tạo</h2>
            <p className="mt-2 text-xs text-slate-400 leading-relaxed">
              Chương trình chẩn đoán bệnh phổi từ ảnh X-quang ngực — phân loại 15 dấu hiệu bệnh lý phổi bằng học sâu.
            </p>
          </div>
        </div>

        {/* Footer brand info */}
        <div className="flex justify-between items-center z-10 text-[11px] text-slate-500">
          <div>© 2026 CheXNet AI System. All rights reserved.</div>
          <div className="flex gap-4">
            <a href="#" className="hover:text-slate-300 transition-colors">Điều khoản</a>
            <a href="#" className="hover:text-slate-300 transition-colors">Bảo mật</a>
          </div>
        </div>
      </div>

      {/* Right side: Modern login card form */}
      <div className="w-full lg:w-5/12 flex items-center justify-center p-6 md:p-12 relative">
        {/* Subtle grid for right side too */}
        <div className="absolute inset-0 bg-medical-grid opacity-[0.03] dark:opacity-[0.015] pointer-events-none" />

        <div className="w-full max-w-md bg-white dark:bg-[#1a1d27] rounded-3xl p-8 shadow-xl shadow-slate-100 dark:shadow-none border border-slate-100 dark:border-slate-800 transition-all duration-300 z-10">
          <div className="text-center mb-8">
            <div className="inline-flex lg:hidden items-center justify-center w-12 h-12 rounded-2xl bg-gradient-to-tr from-[#004ac6] to-[#00d2ff] mb-4">
              <span className="material-symbols-outlined text-white text-[28px]">clinical_notes</span>
            </div>
            <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-400 bg-clip-text text-transparent">
              Chào mừng trở lại!
            </h1>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
              Đăng nhập để tiếp tục chẩn đoán và phân tích hình ảnh X-quang
            </p>
          </div>

          <Spin spinning={isLoading} tip="Đang xác thực thông tin bác sĩ...">
            <Form
              form={form}
              name="login_form"
              layout="vertical"
              onFinish={onFinish}
              requiredMark={false}
              autoComplete="off"
              className="space-y-4"
              initialValues={{ email: "admin@chexnet.vn" }}
            >
              <Form.Item
                label={<span className="text-xs font-semibold text-slate-700 dark:text-slate-300">EMAIL BÁC SĨ</span>}
                name="email"
                rules={[
                  { required: true, message: "Vui lòng nhập email của bạn!" },
                  { type: "email", message: "Định dạng email không hợp lệ!" }
                ]}
              >
                <Input
                  prefix={<span className="material-symbols-outlined text-slate-400 mr-2 text-[18px]">mail</span>}
                  placeholder="admin@chexnet.vn"
                  className="h-11 rounded-xl text-slate-700 dark:text-slate-200 border-slate-200 dark:border-slate-700 hover:border-blue-500 dark:bg-slate-800/40"
                />
              </Form.Item>

              <Form.Item
                label={<span className="text-xs font-semibold text-slate-700 dark:text-slate-300">MẬT KHẨU</span>}
                name="password"
                rules={[{ required: true, message: "Vui lòng nhập mật khẩu của bạn!" }]}
              >
                <Input.Password
                  prefix={<span className="material-symbols-outlined text-slate-400 mr-2 text-[18px]">lock</span>}
                  placeholder="••••••••"
                  className="h-11 rounded-xl text-slate-700 dark:text-slate-200 border-slate-200 dark:border-slate-700 hover:border-blue-500 dark:bg-slate-800/40"
                />
              </Form.Item>

              <div className="flex justify-between items-center text-xs mt-2">
                <label className="flex items-center gap-1.5 cursor-pointer text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200">
                  <input type="checkbox" className="rounded border-slate-300 dark:border-slate-700 text-[#004ac6] focus:ring-[#004ac6]" />
                  <span>Duy trì đăng nhập</span>
                </label>
                <a href="#" className="text-blue-600 dark:text-blue-400 hover:underline font-medium">Quên mật khẩu?</a>
              </div>

              <Form.Item className="pt-2">
                <Button
                  type="primary"
                  htmlType="submit"
                  className="w-full h-11 rounded-xl font-bold tracking-wider bg-gradient-to-r from-[#004ac6] to-[#0082f0] hover:from-[#003da6] hover:to-[#0072d0] border-none shadow-md shadow-blue-500/10 flex items-center justify-center transition-all duration-300 active:scale-[0.98]"
                >
                  ĐĂNG NHẬP HỆ THỐNG
                </Button>
              </Form.Item>
            </Form>
          </Spin>

          <div className="mt-6 rounded-xl border border-slate-100 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-900/40 p-4">
            <p className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-3">
              Tài khoản demo
            </p>
            <div className="flex flex-col gap-2">
              <button
                type="button"
                onClick={() => fillDemoAccount("admin@chexnet.vn", "Admin@123456")}
                className="w-full text-left px-3 py-2 rounded-lg border border-blue-200 dark:border-blue-800/60 bg-white dark:bg-slate-800/60 hover:border-blue-400 transition-colors"
              >
                <span className="text-xs font-bold text-blue-700 dark:text-blue-300">Admin</span>
                <span className="block text-[11px] text-slate-500 dark:text-slate-400 font-mono mt-0.5">
                  admin@chexnet.vn · Admin@123456
                </span>
              </button>
              <button
                type="button"
                onClick={() => fillDemoAccount("bacsi@chexnet.vn", "Doctor@123456")}
                className="w-full text-left px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/60 hover:border-slate-300 dark:hover:border-slate-600 transition-colors"
              >
                <span className="text-xs font-bold text-slate-700 dark:text-slate-200">Bác sĩ</span>
                <span className="block text-[11px] text-slate-500 dark:text-slate-400 font-mono mt-0.5">
                  bacsi@chexnet.vn · Doctor@123456
                </span>
              </button>
            </div>
          </div>

          <div className="mt-6 text-center text-xs text-slate-500 dark:text-slate-400 border-t border-slate-100 dark:border-slate-800 pt-6">
            Chưa có tài khoản bác sĩ?{" "}
            <Link to="/register" className="text-blue-600 dark:text-blue-400 font-semibold hover:underline">
              Đăng ký tài khoản mới
            </Link>
          </div>
        </div>
      </div>
      
      {/* Animated scan helper css style */}
      <style>{`
        @keyframes scan {
          0% { top: 5%; }
          50% { top: 95%; }
          100% { top: 5%; }
        }
      `}</style>
    </div>
  );
};

export default LoginPage;
