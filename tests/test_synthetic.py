import numpy as np

from livecodec import j2k, metrics
from livecodec.baseline import synthetic_volume


def test_budget_encode_hits_target_and_quality_is_monotone():
    vol = synthetic_volume(shape=(16, 128, 128))
    results = []
    for mb in [0.03, 0.1, 0.25]:  # raw volume is ~0.5 MB
        budget = int(mb * 1e6)
        streams, _ = j2k.encode_to_budget(vol, budget)
        actual = sum(len(s) for s in streams)
        assert actual < vol.nbytes
        assert 0.5 < actual / budget < 1.5, f"budget {budget} missed: got {actual}"
        recon = j2k.decode_volume(streams, vol)
        m = metrics.evaluate(vol, recon)
        assert np.isfinite(m["psnr"]), "lossy encode unexpectedly lossless"
        results.append((actual, m["psnr"]))
    sizes, psnrs = zip(*results)
    assert len(set(sizes)) == len(sizes) and list(sizes) == sorted(sizes)
    assert list(psnrs) == sorted(psnrs), f"PSNR not monotone with bytes: {results}"


def test_lossless_roundtrip():
    vol = synthetic_volume(shape=(4, 64, 64))
    streams = j2k.encode_volume(vol, ratio=1.0)
    recon = j2k.decode_volume(streams, vol)
    assert np.array_equal(vol, recon)
