import express, { Request, Response } from 'express';
import cors from 'cors';
import cookieParser from 'cookie-parser';
import path from 'path';
import swaggerUi from 'swagger-ui-express';
import YAML from 'yamljs';
import { trimRequest } from '@common/middlewares/trim.middleware';
import { errorHandler } from '@common/middlewares/error.middleware';
import mainRouter from '@common/routes/index';
import { ALLOWED_ORIGINS } from '@config/environment';
import { RedisAdapter } from '@config/redis';

const app = express();

// Load Swagger document
const swaggerDocument = YAML.load(path.resolve('./swagger.yaml'));

// ── Middleware Chain ──────────────────────────────────────────────────────
app.use(
    cors({
        origin: ALLOWED_ORIGINS,
        credentials: true,
    }),
);
app.use(express.json({ limit: '20mb' }));
app.use(express.urlencoded({ extended: true, limit: '20mb' }));
app.use(cookieParser());
app.use(trimRequest);

// ── Swagger UI Docs ───────────────────────────────────────────────────────
app.use('/docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument));

// ── Static Files (uploads) ───────────────────────────────────────────────
app.use('/uploads', express.static(path.resolve('./uploads')));

// ── Health Check ──────────────────────────────────────────────────────────
app.get('/health', (_req: Request, res: Response) => {
    res.status(200).json({
        status: 'ok',
        service: 'CheXNet V3 Backend',
        version: '1.0.0',
        timestamp: new Date().toISOString(),
        redis: RedisAdapter.isConnected() ? 'connected' : 'unavailable',
    });
});

// ── API Routes ────────────────────────────────────────────────────────────
app.use('/api/v1', mainRouter);

// ── Global Error Handler (phải đăng ký cuối cùng) ───────────────────────
app.use(errorHandler);

export default app;
