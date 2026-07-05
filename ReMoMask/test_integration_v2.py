"""
Integration test: verify ZeRetriever with query projector produces
correct dimensions for SSTA consumption.

Tests:
1. Load ZeRetriever with trained projector
2. Forward pass with test captions
3. Verify re_dict shapes: both re_motion and re_text should be (B, K, 1, 1024)
4. Verify SSTA can consume re_dict without shape errors
"""

import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.rag.ze_retriever import ZeRetriever
from models.transformer.semantics_modulated import SemanticsModulatedAttention


def test_integration(projector_path: str = 'logs/query_projector/best_projector.pt',
                     database_path: str = 'database_ze'):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("Integration Test: ZeRetriever + SSTA dimension alignment")
    print("=" * 60)

    # --- 1. Load ZeRetriever with projector ---
    print("\n[1] Loading ZeRetriever with query projector ...")
    retriever = ZeRetriever(
        database_path=database_path,
        query_projector_path=projector_path,
        top_k=2,
        num_retrieval=10,
        use_shuffle=False,
    )
    retriever = retriever.to(device)

    # --- 2. Forward pass ---
    print("\n[2] Running forward pass ...")
    captions = [
        'a person walks forward and then turns around',
        'a person jumps up and down',
    ]
    re_dict = retriever(captions, k=2)

    re_motion = re_dict['re_motion']
    re_text = re_dict['re_text']

    print(f"  re_motion shape: {re_motion.shape}")
    print(f"  re_text   shape: {re_text.shape}")

    # --- 3. Verify shapes ---
    print("\n[3] Verifying shapes ...")
    B, K = 2, 2
    code_dim = retriever.code_dim  # 1024

    assert re_motion.shape == (B, K, 1, code_dim), \
        f"re_motion shape mismatch: {re_motion.shape} != ({B}, {K}, 1, {code_dim})"
    print(f"  PASS: re_motion is ({B}, {K}, 1, {code_dim})")

    assert re_text.shape == (B, K, 1, code_dim), \
        f"re_text shape mismatch: {re_text.shape} != ({B}, {K}, 1, {code_dim})"
    print(f"  PASS: re_text   is ({B}, {K}, 1, {code_dim})")

    # --- 4. Verify SSTA consumption ---
    print("\n[4] Testing SSTA consumption ...")
    latent_dim = 512
    ssta = SemanticsModulatedAttention(
        latent_dim=latent_dim,
        text_latent_dim=latent_dim,
        num_heads=8,
        dropout=0.1,
        rt_in_value=False,
        retrieval_dim=code_dim,  # 1024
    )
    ssta = ssta.to(device)

    # Create dummy inputs matching typical training shapes
    N_tokens = 49  # typical motion token sequence length
    x = torch.randn(B, N_tokens, latent_dim, device=device)
    xf = torch.randn(B, 1, latent_dim, device=device)
    src_mask = torch.ones(B, N_tokens, 1, device=device)
    cond_type = torch.tensor([[[11]], [[11]]], dtype=torch.float, device=device)

    # Move re_dict to device
    re_dict_dev = {k: v.to(device) for k, v in re_dict.items()}

    try:
        output = ssta(x, xf, src_mask, cond_type, re_dict=re_dict_dev)
        print(f"  PASS: SSTA output shape = {output.shape}")
        assert output.shape == (B, N_tokens, latent_dim), \
            f"SSTA output shape mismatch: {output.shape}"
        print(f"  PASS: Output matches expected ({B}, {N_tokens}, {latent_dim})")
    except RuntimeError as e:
        print(f"  FAIL: SSTA raised RuntimeError: {e}")
        return False

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--projector_path', type=str,
                        default='logs/query_projector/best_projector.pt')
    parser.add_argument('--database_path', type=str, default='database_ze')
    args = parser.parse_args()

    success = test_integration(args.projector_path, args.database_path)
    sys.exit(0 if success else 1)
