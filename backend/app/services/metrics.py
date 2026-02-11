# backend/app/services/metrics.py
"""
Métricas de evaluación para reconstrucción de imágenes.

MÉTRICAS IMPLEMENTADAS:
1. Error L2 (Mean Squared Error)
2. SSIM (Structural Similarity Index)
3. IoU (Intersection over Union) para contornos
4. Hausdorff Distance
5. Frechet Distance

COMPLEJIDAD:
- L2: O(N) donde N = número de píxeles
- SSIM: O(N) con ventanas locales
- IoU: O(N)
- Hausdorff: O(N²) (costoso, usar solo para validación)
- Frechet: O(N²)
"""

import numpy as np
from typing import Tuple, Dict
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import directed_hausdorff

class ImageMetrics:
    """
    Calculador de métricas de similitud entre imágenes.
    
    JUSTIFICACIÓN DE MÉTRICAS:
    - L2: Simple, rápido, penaliza errores grandes
    - SSIM: Correlaciona mejor con percepción humana
    - IoU: Estándar en visión computacional
    - Hausdorff: Útil para contornos, pero costoso
    """
    
    @staticmethod
    def compute_l2_error(
        image1: np.ndarray,
        image2: np.ndarray,
        normalize: bool = True
    ) -> float:
        """
        Error L2 (RMSE) entre dos imágenes.
        
        Args:
            image1, image2: Arrays de forma (H, W) o (H, W, C)
            normalize: Si True, normaliza por rango de valores
        
        Returns:
            Error L2 en rango [0, 1] si normalize=True, sino [0, 255]
        """
        # Asegurar mismo tipo
        img1 = image1.astype(np.float64)
        img2 = image2.astype(np.float64)
        
        # Calcular MSE
        mse = np.mean((img1 - img2) ** 2)
        rmse = np.sqrt(mse)
        
        if normalize:
            # Normalizar por rango máximo posible
            max_val = 255.0 if image1.dtype == np.uint8 else 1.0
            return rmse / max_val
        
        return rmse
    
    @staticmethod
    def compute_ssim(
        image1: np.ndarray,
        image2: np.ndarray,
        multichannel: bool = None
    ) -> float:
        """
        Structural Similarity Index (SSIM).
        
        SSIM ∈ [-1, 1], donde 1 = idénticas
        
        VENTAJA sobre L2:
        - Considera estructura local, no solo diferencias de píxeles
        - Más cercano a percepción humana
        """
        # Auto-detectar multicanal
        if multichannel is None:
            multichannel = len(image1.shape) == 3 and image1.shape[2] > 1
        
        # Convertir a float si es necesario
        img1 = image1.astype(np.float64)
        img2 = image2.astype(np.float64)
        
        if multichannel:
            return ssim(img1, img2, channel_axis=2, data_range=255.0)
        else:
            return ssim(img1, img2, data_range=255.0)
    
    @staticmethod
    def compute_iou(
        mask1: np.ndarray,
        mask2: np.ndarray
    ) -> float:
        """
        Intersection over Union para máscaras binarias.
        
        IoU = |A ∩ B| / |A ∪ B|
        
        Args:
            mask1, mask2: Arrays booleanos o binarios (0/1)
        
        Returns:
            IoU en [0, 1]
        """
        # Convertir a binario
        m1 = mask1.astype(bool)
        m2 = mask2.astype(bool)
        
        intersection = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    @staticmethod
    def compute_hausdorff_distance(
        contour1: np.ndarray,
        contour2: np.ndarray
    ) -> float:
        """
        Distancia de Hausdorff entre dos contornos.
        
        ADVERTENCIA: O(N²) - costoso para contornos grandes
        
        Args:
            contour1, contour2: Arrays de forma (M, 2) con coordenadas (x, y)
        
        Returns:
            Distancia de Hausdorff (máxima distancia mínima)
        """
        d1 = directed_hausdorff(contour1, contour2)[0]
        d2 = directed_hausdorff(contour2, contour1)[0]
        
        return max(d1, d2)
    
    @staticmethod
    def compute_frechet_distance(
        contour1: np.ndarray,
        contour2: np.ndarray
    ) -> float:
        """
        Distancia de Fréchet (similitud de curvas).
        
        COMPLEJIDAD: O(N²) con programación dinámica
        
        INTERPRETACIÓN: Distancia que un perro y su dueño deben
        recorrer si caminan por curvas diferentes sin retroceder.
        """
        def _recursive_frechet(i: int, j: int, memo: dict) -> float:
            """Implementación recursiva con memoización."""
            if (i, j) in memo:
                return memo[(i, j)]
            
            if i == 0 and j == 0:
                result = np.linalg.norm(contour1[0] - contour2[0])
            elif i > 0 and j == 0:
                result = max(
                    _recursive_frechet(i-1, 0, memo),
                    np.linalg.norm(contour1[i] - contour2[0])
                )
            elif i == 0 and j > 0:
                result = max(
                    _recursive_frechet(0, j-1, memo),
                    np.linalg.norm(contour1[0] - contour2[j])
                )
            else:
                result = max(
                    min(
                        _recursive_frechet(i-1, j, memo),
                        _recursive_frechet(i-1, j-1, memo),
                        _recursive_frechet(i, j-1, memo)
                    ),
                    np.linalg.norm(contour1[i] - contour2[j])
                )
            
            memo[(i, j)] = result
            return result
        
        memo = {}
        return _recursive_frechet(len(contour1)-1, len(contour2)-1, memo)
    
    @staticmethod
    def comprehensive_evaluation(
        original_image: np.ndarray,
        reconstructed_image: np.ndarray,
        original_contour: np.ndarray = None,
        reconstructed_contour: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Evaluación completa con todas las métricas.
        
        Returns:
            Diccionario con todas las métricas calculadas
        """
        metrics = {}
        
        # Métricas de imagen completa
        metrics['l2_error'] = ImageMetrics.compute_l2_error(
            original_image, reconstructed_image, normalize=True
        )
        metrics['ssim'] = ImageMetrics.compute_ssim(
            original_image, reconstructed_image
        )
        
        # Métricas de contorno (si están disponibles)
        if original_contour is not None and reconstructed_contour is not None:
            # Máscaras binarias para IoU
            h, w = original_image.shape[:2]
            mask_orig = np.zeros((h, w), dtype=bool)
            mask_recon = np.zeros((h, w), dtype=bool)
            
            # Llenar máscaras (simplificado)
            for point in original_contour:
                y, x = int(point[0]), int(point[1])
                if 0 <= y < h and 0 <= x < w:
                    mask_orig[y, x] = True
            
            for point in reconstructed_contour:
                y, x = int(point[0]), int(point[1])
                if 0 <= y < h and 0 <= x < w:
                    mask_recon[y, x] = True
            
            metrics['iou'] = ImageMetrics.compute_iou(mask_orig, mask_recon)
            
            # Distancias de contorno (submuestrear si es muy grande)
            max_points = 500
            if len(original_contour) > max_points:
                indices = np.linspace(0, len(original_contour)-1, max_points, dtype=int)
                orig_sampled = original_contour[indices]
                recon_sampled = reconstructed_contour[indices]
            else:
                orig_sampled = original_contour
                recon_sampled = reconstructed_contour
            
            metrics['hausdorff'] = ImageMetrics.compute_hausdorff_distance(
                orig_sampled, recon_sampled
            )
            
            # Fréchet solo para contornos pequeños (muy costoso)
            if len(orig_sampled) < 200 and len(recon_sampled) < 200: # Added check for recon_sampled as well
                metrics['frechet'] = ImageMetrics.compute_frechet_distance(
                    orig_sampled, recon_sampled
                )
        
        return metrics