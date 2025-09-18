try:
    from .model_encoder import VisualEncoder2D, VisualEncoder3D, Aggregator
    from .utils import MLP
    from .cross_attention import CrossAttentionLayer as CrossAttention
    from .model_decoder import Decoder
except ImportError as e:
    print("[models] Relative import failed:", e)
    print("[models] Trying absolute import fallback...")
    try:
        from model_encoder import VisualEncoder2D, VisualEncoder3D, Aggregator
        from utils import MLP
        from cross_attention import CrossAttentionLayer as CrossAttention
        from model_decoder import Decoder
    except ImportError as e2:
        print("[models] Absolute import also failed:", e2)
        raise e2
