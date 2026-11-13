import type { ImgHTMLAttributes } from 'react';
import { useScanImage } from '../../hooks/useScanImage';

export default function ScanImage({ src, alt, ...props }: ImgHTMLAttributes<HTMLImageElement>) {
    const { url: image, error } = useScanImage(src);
    if (error) return <span role="alert">Không tải được ảnh. Vui lòng mở lại lần chụp.</span>;
    if (src && !image) return <span role="status">Đang tải ảnh xem lại…</span>;
    return <img {...props} src={image} alt={alt} />;
}
