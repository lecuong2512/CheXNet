import { Server as HttpServer } from 'http';
import { Server } from 'socket.io';
import { verifyToken } from '@common/utils/token.utils';
import { logger } from '@common/utils/logger';
import { ALLOWED_ORIGINS } from '@config/environment';

class SocketService {
    private io: Server | null = null;

    public initialize(httpServer: HttpServer): void {
        this.io = new Server(httpServer, {
            cors: {
                origin: ALLOWED_ORIGINS,
                credentials: true,
            },
        });

        // Middleware xác thực socket
        this.io.use((socket, next) => {
            try {
                const token = socket.handshake.auth?.token || socket.handshake.query?.token as string;
                if (!token) {
                    return next(new Error('Không có token xác thực'));
                }
                const decoded = verifyToken(token);
                (socket as any).user = decoded;
                next();
            } catch {
                next(new Error('Token không hợp lệ'));
            }
        });

        this.io.on('connection', (socket) => {
            const user = (socket as any).user;
            logger.info(`🔌 Socket kết nối: userId=${user?.userId}`);

            // Join rooms tự động theo userId và role
            if (user?.userId) {
                socket.join(`u_${user.userId}`);
                socket.join(`role_${user.role?.toUpperCase()}`);
            }

            socket.on('join:patient', (patientId: string) => {
                socket.join(`patient_${patientId}`);
            });

            socket.on('leave:patient', (patientId: string) => {
                socket.leave(`patient_${patientId}`);
            });

            socket.on('disconnect', () => {
                logger.info(`🔌 Socket ngắt kết nối: userId=${user?.userId}`);
            });
        });

        logger.info('✅ Socket.IO đã khởi động');
    }

    /** Gửi sự kiện đến một user cụ thể */
    public sendNotification(userId: string, event: string, data: unknown): void {
        this.io?.to(`u_${userId}`).emit(event, data);
    }

    /** Gửi sự kiện đến tất cả user có role cụ thể */
    public sendToRole(roleCode: string, event: string, data: unknown): void {
        this.io?.to(`role_${roleCode.toUpperCase()}`).emit(event, data);
    }

    /** Gửi sự kiện đến tất cả client trong room */
    public sendToRoom(room: string, event: string, data: unknown): void {
        this.io?.to(room).emit(event, data);
    }

    /** Broadcast đến tất cả (dùng cho admin notifications) */
    public broadcast(event: string, data: unknown): void {
        this.io?.emit(event, data);
    }
}

// Export singleton instance
export const socketService = new SocketService();
