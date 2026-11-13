export const isDicomFile = (file: File) => /\.(dcm|dicom)$/i.test(file.name) ||
    ['application/dicom', 'image/dicom'].includes(file.type);

export const isSupportedScan = (file: File) => isDicomFile(file) ||
    ['image/png', 'image/jpeg', 'image/jpg'].includes(file.type);
