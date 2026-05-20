import numpy as np

def compute_disparity_block_matching(I_l, I_r, window_size=5, method='SAD', max_disparity=64):
    """
    Compute disparity map using block matching
    """
    height, width = I_l.shape
    k = window_size // 2
    disparity_map = np.zeros_like(I_l, dtype=np.float32)
    
    I_l = I_l.astype(np.float32)
    I_r = I_r.astype(np.float32)
    
    for y in range(k, height - k):
        for x in range(max_disparity + k, width - k):
            best_cost = float('inf')
            best_d = 0
            for d in range(max_disparity):
                if x - d - k < 0: continue
                cost = 0
                for dx in range(-k, k + 1):
                    for dy in range(-k, k + 1):
                        val_l = I_l[y + dy, x + dx]
                        val_r = I_r[y + dy, x + dx - d]
                        if method == 'SAD': cost += abs(val_l - val_r)
                        else: cost += (val_l - val_r) ** 2
                
                if cost < best_cost:
                    best_cost = cost
                    best_d = d
            disparity_map[y, x] = best_d
    return disparity_map

def compute_disparity_dp(I_l, I_r, sigma=2.0, c0=1.0):
    """
    Compute disparity map using dynamic programming scanline optimization.
    """
    I_l = I_l.astype(np.float32)
    I_r = I_r.astype(np.float32)
    height, width = I_l.shape
    disp_left = np.zeros((height, width), dtype=np.float32)
    
    for y in range(height):
        Il_row = I_l[y, :]
        Ir_row = I_r[y, :]
        N = width
        diff = Il_row[:, None] - Ir_row[None, :]
        d = (diff * diff) / (sigma * sigma)
        
        D = np.zeros((N + 1, N + 1), dtype=np.float32)
        for i in range(1, N + 1): D[i, 0] = i * c0
        for j in range(1, N + 1): D[0, j] = j * c0
        
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                cost_match = D[i - 1, j - 1] + d[i - 1, j - 1]
                cost_skip_left = D[i - 1, j] + c0
                cost_skip_right = D[i, j - 1] + c0
                D[i, j] = min(cost_match, cost_skip_left, cost_skip_right)
                
        i, j = N, N
        eps = 1e-6
        while i > 0 or j > 0:
            if i > 0 and j > 0 and abs(D[i, j] - (D[i - 1, j - 1] + d[i - 1, j - 1])) <= eps:
                disp_left[y, i - 1] = abs((i - 1) - (j - 1))
                i -= 1; j -= 1
            elif i > 0 and abs(D[i, j] - (D[i - 1, j] + c0)) <= eps:
                disp_left[y, i - 1] = 0.0
                i -= 1
            else:
                j -= 1
    return disp_left

def normalize_disparity(disparity_map):
    valid_pixels = disparity_map[disparity_map > 0]
    if len(valid_pixels) > 0:
        min_val = np.min(valid_pixels)
        max_val = np.max(valid_pixels)
        if max_val > min_val:
            normalized = (disparity_map - min_val) / (max_val - min_val) * 255
            return normalized.astype(np.uint8)
    return np.zeros_like(disparity_map, dtype=np.uint8)
