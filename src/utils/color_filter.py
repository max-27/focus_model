from typing import Any, Tuple, List
import cv2
import numpy as np


# upper and lower purple ranges in HSV color space
LOWER_HSV_RANGE = np.array([130,20,20])
UPPER_HSV_RANGE = np.array([180,255,255])


class ColorFilter:
    def __init__(
                self, 
                ) -> None:
        """Color filter to exclude images containing no or not enough tissue

        Parameters
        ----------
        threshold: float
            Ratio of pixels that correspond to tissue in image
        data_out_type: ImageFormats
            Enum defining final data type of filtered images for storage
        """
        self.counter_img_keep = 0
        self.counter_img_disregard = 0
    
    def _read_img(self, dir: str) -> np.array:
        return cv2.imread(dir)

    def _convert_BGR_to_HSV(self, img: np.array) -> np.array:
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def _get_mask(self, img_hsv: np.array) -> np.array:
        return cv2.inRange(img_hsv, LOWER_HSV_RANGE, UPPER_HSV_RANGE)

    def _get_filtered_img(self, img: np.array, mask: np.array) -> np.array:
        return cv2.bitwise_and(img, img, mask=mask)
    
    def _filter(self, img: np.array) -> np.array:
        """Filter pipeline:
        1. Convert to HSV space
        2. Retrieve image mask
        3. Retrieve filtered image
        
        Parameters
        ----------
        img: np.array
            Input image
        
        Returns
        -------
        np.array
            Filtered image
        """
        hsv_img = self._convert_BGR_to_HSV(img)
        mask = self._get_mask(hsv_img)
        return self._get_filtered_img(img, mask)
    
    def _get_sample_ratio(self, img_arr: np.array) -> float:
        """Retrieves the ratio of the image containing sample
        
        Parameters
        ----------
        img_arr: np.array
            Input image
        
        Returns
        -------
        float
            Ratio of the image containing sample
        """
        hsv_img = self._convert_BGR_to_HSV(img_arr)
        mask = self._get_mask(hsv_img)
        sample_pixel_count = (mask/255).astype(int).sum()
        return (sample_pixel_count/mask.size)*100