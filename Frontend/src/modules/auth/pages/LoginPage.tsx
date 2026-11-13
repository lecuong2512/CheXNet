import React from "react";
import AuthLayout from "../../../components/common/AuthLayout";
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

  return <AuthLayout>
    <div className="auth-heading">
      <h1>Đăng nhập</h1>
      <p>Nhập thông tin tài khoản để tiếp tục.</p>
    </div>
    <Spin spinning={isLoading} description="Đang đăng nhập…">
      <Form form={form} name="login_form" layout="vertical" onFinish={onFinish}
        requiredMark={false} initialValues={{ email: 'admin@chexnet.vn' }}>
        <Form.Item label="Email" name="email" rules={[
          { required: true, message: 'Vui lòng nhập email.' },
          { type: 'email', message: 'Email chưa đúng định dạng.' },
        ]}>
          <Input placeholder="Nhập email" autoComplete="username" />
        </Form.Item>
        <Form.Item label="Mật khẩu" name="password" rules={[{ required: true, message: 'Vui lòng nhập mật khẩu.' }]}>
          <Input.Password placeholder="Nhập mật khẩu" autoComplete="current-password" />
        </Form.Item>
        <div className="auth-forgot">
          <button type="button" onClick={() => message.info('Vui lòng liên hệ quản trị viên để được hỗ trợ đặt lại mật khẩu.')}>Quên mật khẩu?</button>
        </div>
        <Button type="primary" htmlType="submit" block loading={isLoading}>Đăng nhập</Button>
      </Form>
    </Spin>
    <p className="auth-switch">Chưa có tài khoản? <Link to="/register">Đăng ký tài khoản</Link></p>
    <details className="auth-demo">
      <summary>Tài khoản dùng thử</summary>
      <div className="auth-demo-accounts">
        <button type="button" onClick={() => fillDemoAccount('admin@chexnet.vn', 'Admin@123456')}>
          <strong>Quản trị viên</strong><span>admin@chexnet.vn</span><span>Admin@123456</span>
        </button>
        <button type="button" onClick={() => fillDemoAccount('bacsi@chexnet.vn', 'Doctor@123456')}>
          <strong>Bác sĩ</strong><span>bacsi@chexnet.vn</span><span>Doctor@123456</span>
        </button>
      </div>
    </details>
  </AuthLayout>;
};

export default LoginPage;
