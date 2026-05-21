import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { useDispatch } from "react-redux";
import { Form, Input, Select, Button, message, Spin } from "antd";
import { useRegisterMutation } from "../../../stores/baseApi";
import { setCredentials } from "../../../stores/authSlice";

const { Option } = Select;

const RegisterPage: React.FC = () => {
  const [form] = Form.useForm();
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const [register, { isLoading }] = useRegisterMutation();

  const onFinish = async (values: any) => {
    const { name, email, password, department } = values;
    try {
      const response = await register({ name, email, password, department }).unwrap();
      if (response?.success && response?.data) {
        const { user, accessToken, refreshToken } = response.data;
        dispatch(setCredentials({ user, accessToken, refreshToken }));
        message.success({
          content: "Đăng ký thành công! Chào mừng Bác sĩ " + user.name,
          duration: 3,
        });
        navigate("/");
      } else {
        message.error("Đăng ký thất bại. Vui lòng thử lại.");
      }
    } catch (err: any) {
      const errMsg = err?.data?.message || err?.message || "Đăng ký thất bại. Vui lòng kiểm tra lại thông tin.";
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

        {/* Interactive Deep Neural Net Visualizer */}
        <div className="my-auto flex flex-col items-center justify-center relative z-10">
          <div className="relative w-[360px] h-[360px] border border-blue-500/20 bg-slate-900/40 rounded-3xl p-6 shadow-2xl flex items-center justify-center overflow-hidden backdrop-blur-md">
            {/* Medical scanning grid overlay */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(0,74,198,0.05)_1px,transparent_1px),linear-gradient(to_bottom,rgba(0,74,198,0.05)_1px,transparent_1px)] bg-[size:16px_16px]" />

            {/* Neural Net Nodes Graphic */}
            <svg width="240" height="240" viewBox="0 0 240 240" fill="none" className="opacity-80 drop-shadow-[0_0_10px_rgba(0,210,255,0.25)]">
              {/* Connection lines */}
              <line x1="40" y1="60" x2="120" y2="40" stroke="#00d2ff" strokeWidth="1.5" opacity="0.3" />
              <line x1="40" y1="60" x2="120" y2="120" stroke="#00d2ff" strokeWidth="1.5" opacity="0.3" />
              <line x1="40" y1="120" x2="120" y2="40" stroke="#00d2ff" strokeWidth="1.5" opacity="0.3" />
              <line x1="40" y1="120" x2="120" y2="120" stroke="#00d2ff" strokeWidth="1.5" opacity="0.3" />
              <line x1="40" y1="120" x2="120" y2="200" stroke="#00d2ff" strokeWidth="1.5" opacity="0.3" />
              <line x1="40" y1="180" x2="120" y2="120" stroke="#00d2ff" strokeWidth="1.5" opacity="0.3" />
              <line x1="40" y1="180" x2="120" y2="200" stroke="#00d2ff" strokeWidth="1.5" opacity="0.3" />
              
              <line x1="120" y1="40" x2="200" y2="90" stroke="#10b981" strokeWidth="1.5" opacity="0.4" />
              <line x1="120" y1="120" x2="200" y2="90" stroke="#10b981" strokeWidth="1.5" opacity="0.4" />
              <line x1="120" y1="120" x2="200" y2="150" stroke="#10b981" strokeWidth="1.5" opacity="0.4" />
              <line x1="120" y1="200" x2="200" y2="150" stroke="#10b981" strokeWidth="1.5" opacity="0.4" />

              {/* Layer 1 Nodes */}
              <circle cx="40" cy="60" r="8" fill="#0c1033" stroke="#00d2ff" strokeWidth="2.5" />
              <circle cx="40" cy="120" r="8" fill="#0c1033" stroke="#00d2ff" strokeWidth="2.5" />
              <circle cx="40" cy="180" r="8" fill="#0c1033" stroke="#00d2ff" strokeWidth="2.5" />
              
              {/* Layer 2 Nodes */}
              <circle cx="120" cy="40" r="8" fill="#0c1033" stroke="#00d2ff" strokeWidth="2.5" />
              <circle cx="120" cy="120" r="10" fill="#0c1033" stroke="#00d2ff" strokeWidth="3" className="animate-[pulse_1.5s_infinite]" />
              <circle cx="120" cy="200" r="8" fill="#0c1033" stroke="#00d2ff" strokeWidth="2.5" />
              
              {/* Layer 3 (Output Nodes) */}
              <circle cx="200" cy="90" r="8" fill="#0c1033" stroke="#10b981" strokeWidth="2.5" />
              <circle cx="200" cy="150" r="8" fill="#0c1033" stroke="#ef4444" strokeWidth="2.5" />

              <text x="35" y="63" fill="#00d2ff" fontSize="8" fontWeight="bold" textAnchor="middle">IN</text>
              <text x="120" y="123" fill="#00d2ff" fontSize="8" fontWeight="bold" textAnchor="middle">AI</text>
              <text x="200" y="93" fill="#10b981" fontSize="8" fontWeight="bold" textAnchor="middle">OK</text>
              <text x="200" y="153" fill="#ef4444" fontSize="8" fontWeight="bold" textAnchor="middle">X</text>
            </svg>
          </div>
          
          <div className="mt-8 text-center max-w-sm">
            <h2 className="text-xl font-bold text-slate-100">Đăng ký Tài khoản Bác sĩ</h2>
            <p className="mt-2 text-xs text-slate-400 leading-relaxed">
              Trở thành thành viên để số hóa hồ sơ bệnh nhân, hỗ trợ ra quyết định lâm sàng tức thời từ các mô hình chẩn đoán AI hàng đầu.
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

      {/* Right side: Modern register card form */}
      <div className="w-full lg:w-5/12 flex items-center justify-center p-6 md:p-12 relative overflow-y-auto h-full">
        {/* Subtle grid for right side too */}
        <div className="absolute inset-0 bg-medical-grid opacity-[0.03] dark:opacity-[0.015] pointer-events-none" />

        <div className="w-full max-w-md bg-white dark:bg-[#1a1d27] rounded-3xl p-8 shadow-xl shadow-slate-100 dark:shadow-none border border-slate-100 dark:border-slate-800 transition-all duration-300 z-10 my-8">
          <div className="text-center mb-6">
            <div className="inline-flex lg:hidden items-center justify-center w-12 h-12 rounded-2xl bg-gradient-to-tr from-[#004ac6] to-[#00d2ff] mb-4">
              <span className="material-symbols-outlined text-white text-[28px]">clinical_notes</span>
            </div>
            <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-400 bg-clip-text text-transparent">
              Đăng ký Bác sĩ
            </h1>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
              Khởi tạo tài khoản chẩn đoán hình ảnh chuyên nghiệp
            </p>
          </div>

          <Spin spinning={isLoading} tip="Đang thiết lập hồ sơ bác sĩ...">
            <Form
              form={form}
              name="register_form"
              layout="vertical"
              onFinish={onFinish}
              requiredMark={false}
              autoComplete="off"
              className="space-y-3"
            >
              <Form.Item
                label={<span className="text-xs font-semibold text-slate-700 dark:text-slate-300">HỌ VÀ TÊN BÁC SĨ</span>}
                name="name"
                rules={[
                  { required: true, message: "Vui lòng nhập họ và tên của bạn!" },
                  { min: 3, message: "Họ tên phải có ít nhất 3 ký tự!" }
                ]}
              >
                <Input
                  prefix={<span className="material-symbols-outlined text-slate-400 mr-2 text-[18px]">person</span>}
                  placeholder="Bác sĩ Nguyễn Văn A"
                  className="h-11 rounded-xl text-slate-700 dark:text-slate-200 border-slate-200 dark:border-slate-700 hover:border-blue-500 dark:bg-slate-800/40"
                />
              </Form.Item>

              <Form.Item
                label={<span className="text-xs font-semibold text-slate-700 dark:text-slate-300">EMAIL CÁ NHÂN / BỆNH VIỆN</span>}
                name="email"
                rules={[
                  { required: true, message: "Vui lòng nhập email!" },
                  { type: "email", message: "Định dạng email không hợp lệ!" }
                ]}
              >
                <Input
                  prefix={<span className="material-symbols-outlined text-slate-400 mr-2 text-[18px]">mail</span>}
                  placeholder="doctor.a@hospital.org"
                  className="h-11 rounded-xl text-slate-700 dark:text-slate-200 border-slate-200 dark:border-slate-700 hover:border-blue-500 dark:bg-slate-800/40"
                />
              </Form.Item>

              <Form.Item
                label={<span className="text-xs font-semibold text-slate-700 dark:text-slate-300">CHUYÊN KHOA / PHÒNG BAN</span>}
                name="department"
                rules={[{ required: true, message: "Vui lòng chọn chuyên khoa!" }]}
              >
                <Select
                  placeholder="Chọn khoa công tác"
                  className="h-11 rounded-xl text-slate-700 dark:text-slate-200 border-slate-200 dark:border-slate-700 hover:border-blue-500 dark:bg-slate-800/40"
                  size="large"
                  dropdownClassName="dark:bg-slate-800"
                >
                  <Option value="Chẩn đoán hình ảnh">Chẩn đoán hình ảnh (Radiology)</Option>
                  <Option value="Khoa Hô hấp">Khoa Hô hấp (Pulmonology)</Option>
                  <Option value="Khoa Cấp cứu">Khoa Cấp cứu (Emergency Room)</Option>
                  <Option value="Khoa Khám bệnh">Khoa Khám bệnh</Option>
                  <Option value="Nghiên cứu khoa học">Nghiên cứu & Đào tạo</Option>
                </Select>
              </Form.Item>

              <Form.Item
                label={<span className="text-xs font-semibold text-slate-700 dark:text-slate-300">MẬT KHẨU (TỐI THIỂU 6 KÝ TỰ)</span>}
                name="password"
                rules={[
                  { required: true, message: "Vui lòng nhập mật khẩu!" },
                  { min: 6, message: "Mật khẩu phải có ít nhất 6 ký tự!" }
                ]}
              >
                <Input.Password
                  prefix={<span className="material-symbols-outlined text-slate-400 mr-2 text-[18px]">lock</span>}
                  placeholder="••••••••"
                  className="h-11 rounded-xl text-slate-700 dark:text-slate-200 border-slate-200 dark:border-slate-700 hover:border-blue-500 dark:bg-slate-800/40"
                />
              </Form.Item>

              <Form.Item
                label={<span className="text-xs font-semibold text-slate-700 dark:text-slate-300">XÁC NHẬN MẬT KHẨU</span>}
                name="confirmPassword"
                dependencies={["password"]}
                rules={[
                  { required: true, message: "Vui lòng xác nhận mật khẩu!" },
                  ({ getFieldValue }) => ({
                    validator(_, value) {
                      if (!value || getFieldValue("password") === value) {
                        return Promise.resolve();
                      }
                      return Promise.reject(new Error("Mật khẩu xác nhận không khớp!"));
                    },
                  }),
                ]}
              >
                <Input.Password
                  prefix={<span className="material-symbols-outlined text-slate-400 mr-2 text-[18px]">lock</span>}
                  placeholder="••••••••"
                  className="h-11 rounded-xl text-slate-700 dark:text-slate-200 border-slate-200 dark:border-slate-700 hover:border-blue-500 dark:bg-slate-800/40"
                />
              </Form.Item>

              <Form.Item className="pt-2">
                <Button
                  type="primary"
                  htmlType="submit"
                  className="w-full h-11 rounded-xl font-bold tracking-wider bg-gradient-to-r from-[#004ac6] to-[#0082f0] hover:from-[#003da6] hover:to-[#0072d0] border-none shadow-md shadow-blue-500/10 flex items-center justify-center transition-all duration-300 active:scale-[0.98]"
                >
                  ĐĂNG KÝ TÀI KHOẢN BÁC SĨ
                </Button>
              </Form.Item>
            </Form>
          </Spin>

          <div className="mt-6 text-center text-xs text-slate-500 dark:text-slate-400 border-t border-slate-100 dark:border-slate-800 pt-4">
            Đã có tài khoản chẩn đoán?{" "}
            <Link to="/login" className="text-blue-600 dark:text-blue-400 font-semibold hover:underline">
              Đăng nhập ngay
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RegisterPage;
