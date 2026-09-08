class CosmeticAIError(Exception):
    """Base exception for CosmeticAI"""
    pass

class ImagePreprocessingError(CosmeticAIError):
    """Raised when image preprocessing fails"""
    pass

class LayoutDetectionError(CosmeticAIError):
    """Raised when YOLO layout detection fails"""
    pass

class OCRError(CosmeticAIError):
    """Raised when OCR text extraction fails"""
    pass

class IngredientDBError(CosmeticAIError):
    """Raised when database loading or querying fails"""
    pass

class LLMServiceError(CosmeticAIError):
    """Raised when LLM analysis fails"""
    pass
