import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";
import { Result, Button } from "antd";

interface Props {
  children?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error:", error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center min-h-screen bg-gray-50 dark:bg-[#1a1d27]">
          <Result
            status="500"
            title={<span className="text-gray-800 dark:text-gray-100 font-bold">Đã có lỗi không mong muốn xảy ra</span>}
            subTitle={<span className="text-gray-500 dark:text-gray-400">Giao diện gặp sự cố khi hiển thị. Vui lòng thử lại.</span>}
            extra={
              <Button type="primary" onClick={() => window.location.reload()} className="h-10 px-6 rounded-xl font-bold bg-[#004ac6] border-none">
                Tải lại trang
              </Button>
            }
          />
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
