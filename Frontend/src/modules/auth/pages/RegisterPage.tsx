import React from "react";
import AuthLayout from "../../../components/common/AuthLayout";
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

  return <AuthLayout>
    <div className="auth-heading">
      <h1>Đăng ký tài khoản</h1>
      <p>Điền thông tin để tạo tài khoản sử dụng hệ thống.</p>
    </div>
    <Spin spinning={isLoading} description="Đang tạo tài khoản…">
      <Form form={form} name="register_form" layout="vertical" onFinish={onFinish} requiredMark={false}>
        <Form.Item label="Họ và tên" name="name" rules={[
          { required: true, message: 'Vui lòng nhập họ và tên.' },
          { min: 3, message: 'Họ tên phải có ít nhất 3 ký tự.' },
        ]}>
          <Input placeholder="Nhập họ và tên" autoComplete="name" />
        </Form.Item>
        <Form.Item label="Email" name="email" rules={[
          { required: true, message: 'Vui lòng nhập email.' },
          { type: 'email', message: 'Email chưa đúng định dạng.' },
        ]}>
          <Input placeholder="Nhập email" autoComplete="email" />
        </Form.Item>
        <Form.Item label="Chuyên khoa / Phòng ban" name="department" rules={[{ required: true, message: 'Vui lòng chọn chuyên khoa.' }]}>
          <Select placeholder="Chọn khoa công tác">
            <Option value="Chẩn đoán hình ảnh">Chẩn đoán hình ảnh</Option>
            <Option value="Khoa Hô hấp">Khoa Hô hấp</Option>
            <Option value="Khoa Cấp cứu">Khoa Cấp cứu</Option>
            <Option value="Khoa Khám bệnh">Khoa Khám bệnh</Option>
            <Option value="Nghiên cứu khoa học">Nghiên cứu & Đào tạo</Option>
          </Select>
        </Form.Item>
        <Form.Item label="Mật khẩu" name="password" rules={[
          { required: true, message: 'Vui lòng nhập mật khẩu.' },
          { min: 6, message: 'Mật khẩu phải có ít nhất 6 ký tự.' },
        ]} extra="Tối thiểu 6 ký tự.">
          <Input.Password placeholder="Nhập mật khẩu" autoComplete="new-password" />
        </Form.Item>
        <Form.Item label="Xác nhận mật khẩu" name="confirmPassword" dependencies={['password']} rules={[
          { required: true, message: 'Vui lòng xác nhận mật khẩu.' },
          ({ getFieldValue }) => ({ validator(_, value) {
            return !value || getFieldValue('password') === value ? Promise.resolve() : Promise.reject(new Error('Mật khẩu xác nhận không khớp.'));
          } }),
        ]}>
          <Input.Password placeholder="Nhập lại mật khẩu" autoComplete="new-password" />
        </Form.Item>
        <Button type="primary" htmlType="submit" block loading={isLoading}>Tạo tài khoản</Button>
      </Form>
    </Spin>
    <p className="auth-switch">Đã có tài khoản? <Link to="/login">Đăng nhập</Link></p>
  </AuthLayout>;
};

export default RegisterPage;
