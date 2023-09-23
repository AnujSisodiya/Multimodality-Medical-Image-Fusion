import numpy as np
import cv2
import pywt

# Our methods
def convertToIntList(arr):
    result = []
    for q in arr.strip(']][[').split('],['):
        x = []
        for i in q.split(','):
            x.append(int(i, 10))
        result.append(x)
    return result

def procrustes(X, Y, scaling=True, reflection='best'):
    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros((n, m - my))), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY

        # standardized distance between X and b*Y*T + c
        d = 1 - traceTA ** 2
        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)
    # rot =1
    # scale=2
    # translate=3
    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform


# Load CT Image
ct_image = cv2.imread('ct_image.jpg', 0)
# Load MRI Registered Image
mri_registered = cv2.imread('mri_registered.jpg', 0)

# Define corresponding points
ctCoord = convertToIntList(input("Enter CT coordinates: "))
mriCoord = convertToIntList(input("Enter MRI coordinates: "))

# Perform image registration
X_pts = np.asarray(ctCoord)
Y_pts = np.asarray(mriCoord)

d, Z_pts, Tform = procrustes(X_pts, Y_pts)
R = np.eye(3)
R[0:2, 0:2] = Tform['rotation']

S = np.eye(3) * Tform['scale']
S[2, 2] = 1
t = np.eye(3)
t[0:2, 2] = Tform['translation']
M = np.dot(np.dot(R, S), t.T).T

# Apply registration transformation to MRI registered image
mri_registered_transformed = cv2.warpAffine(mri_registered, M[0:2, :], (ct_image.shape[1], ct_image.shape[0]))

# DWT Fusion
def dwt_fusion(img1, img2):
    # Perform wavelet transform on both images
    coeffs1 = pywt.dwt2(img1, 'haar')
    coeffs2 = pywt.dwt2(img2, 'haar')

    # Extract approximation coefficients from both images
    appr_coeffs1, (hcoeffs1, vcoeffs1, dcoeffs1) = coeffs1
    appr_coeffs2, (hcoeffs2, vcoeffs2, dcoeffs2) = coeffs2

    # Fuse the approximation coefficients
    fused_appr_coeffs = (appr_coeffs1 + appr_coeffs2) / 2

    # Fuse the horizontal, vertical, and diagonal details coefficients using maximum fusion
    fused_hcoeffs = np.maximum(hcoeffs1, hcoeffs2)
    fused_vcoeffs = np.maximum(vcoeffs1, vcoeffs2)
    fused_dcoeffs = np.maximum(dcoeffs1, dcoeffs2)

    # Combine the fused coefficients
    fused_coeffs = fused_appr_coeffs, (fused_hcoeffs, fused_vcoeffs, fused_dcoeffs)

    # Perform inverse wavelet transform
    fused_img = pywt.idwt2(fused_coeffs, 'haar')

    return fused_img.astype(np.uint8)


# Pixel-based Fusion
def pixel_fusion(img1, img2):
    # Normalize the input images
    img1_norm = img1 / 255.0
    img2_norm = img2 / 255.0

    # Perform pixel-based fusion
    fused_img = (img1_norm + img2_norm) / 2.0

    # Clip the fused image values to the valid range [0, 1]
    fused_img = np.clip(fused_img, 0, 1)

    # Scale the fused image to the range [0, 255]
    fused_img = (fused_img * 255).astype(np.uint8)

    return fused_img

def calculate_psnr(original_img, fused_img):
    # Ensure both images have the same data type for accurate calculations
    original_img = original_img.astype(np.float32)
    fused_img = fused_img.astype(np.float32)

    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((original_img - fused_img) ** 2)

    # Calculate the maximum possible pixel value
    max_pixel_val = np.max(original_img)

    # Calculate the PSNR
    psnr = 10 * np.log10((max_pixel_val ** 2) / mse)

    return psnr

# Display original CT image
cv2.imshow('CT Image', ct_image)

# Display original MRI registered image
cv2.imshow('MRI Registered', mri_registered_transformed)

# Perform pixel-based fusion of MRI registered image and CT scan image
pixel_fused_img = pixel_fusion(mri_registered_transformed, ct_image)

# Perform DWT fusion of MRI registered image and CT scan image
dwt_fused_img = dwt_fusion(mri_registered_transformed, ct_image)



# Display and save the fused images
cv2.imshow('Pixel-based Fusion', pixel_fused_img)
cv2.imshow('DWT Fusion', dwt_fused_img)
cv2.imwrite('pixel_fused_image.jpg', pixel_fused_img)
cv2.imwrite('dwt_fused_image.jpg', dwt_fused_img)

# Load the original CT or MRI image
original_image = cv2.imread('ct_image.jpg', 0)

# Load the fused image obtained using DWT or pixel-based fusion
fused_image = cv2.imread('dwt_fused_image.jpg', 0)

# Calculate PSNR for DWT fused image
psnr_dwt_fused = calculate_psnr(original_image, dwt_fused_img)
print(f"PSNR value for DWT Fusion: {psnr_dwt_fused:.2f} dB")

# Calculate PSNR for pixel-based fused image
psnr_pixel_fused = calculate_psnr(original_image, pixel_fused_img)
print(f"PSNR value for Pixel-based Fusion: {psnr_pixel_fused:.2f} dB")

cv2.waitKey(0)
cv2.destroyAllWindows()
