import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

DXGI_FORMAT_BC1_UNORM = 71
DXGI_FORMAT_BC1_UNORM_SRGB = 72

class EndpointExtractor:
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_dds_bc1_endpoints(self, dds_path: Path) -> Dict[str, Any]:
        """Parse a BC1 DDS and return endpoints_rgb565 (N,2) in row-major block order."""
        data = dds_path.read_bytes()
        if len(data) < 128 or data[0:4] != b"DDS ":
            raise ValueError(f"Not a valid DDS file (missing DDS magic): {dds_path.name}")

        header = data[4:4 + 124]
        if struct.unpack_from("<I", header, 0)[0] != 124:
            raise ValueError("Unexpected DDS header size.")

        height = struct.unpack_from("<I", header, 8)[0]
        width = struct.unpack_from("<I", header, 12)[0]

        ddspf_off = 72
        fourcc = header[ddspf_off + 8: ddspf_off + 12]

        offset = 4 + 124
        if fourcc == b"DXT1":
            pass
        elif fourcc == b"DX10":
            dx10 = data[offset:offset + 20]
            dxgi_format = struct.unpack_from("<I", dx10, 0)[0]
            if dxgi_format not in (DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC1_UNORM_SRGB):
                raise ValueError(f"DDS DX10 format is not BC1 (dxgiFormat={dxgi_format}).")
            offset += 20
        else:
            raise ValueError(f"Unsupported DDS FourCC for BC1 extraction: {fourcc!r}")

        blocks_x = (width + 3) // 4
        blocks_y = (height + 3) // 4
        num_blocks = blocks_x * blocks_y

        needed = offset + num_blocks * 8
        if len(data) < needed:
            raise ValueError("DDS truncated: not enough BC1 blocks.")

        # Vectorized: read all BC1 blocks at once (8 bytes each: 2B c0, 2B c1, 4B indices)
        raw = np.frombuffer(data, dtype=np.uint16, offset=offset, count=num_blocks * 4)
        c0_arr = raw[0::4].astype(np.int32)
        c1_arr = raw[1::4].astype(np.int32)

        endpoints_rgb565 = np.stack([c0_arr, c1_arr], axis=-1)  # (N, 2)

        return {
            "width": int(width),
            "height": int(height),
            "blocks_x": int(blocks_x),
            "blocks_y": int(blocks_y),
            "endpoints_rgb565": endpoints_rgb565,
        }

    def _rgb565_to_q01_vec(self, c: np.ndarray) -> np.ndarray:
        r5 = ((c >> 11) & 0x1F).astype(np.float32) / 31.0
        g6 = ((c >> 5) & 0x3F).astype(np.float32) / 63.0
        b5 = (c & 0x1F).astype(np.float32) / 31.0
        return np.stack([r5, g6, b5], axis=-1)  # (N, 3)

    def extract(self, include_meta: bool = True) -> str:
        """Processes all DDS files in source_dir and writes the dataset."""
        dds_files = sorted(list(self.source_dir.glob("*.dds")))
        if not dds_files:
            raise FileNotFoundError(f"No .dds files found in {self.source_dir}")

        refs = []
        texture_names = []
        
        w0, h0, bx0, by0 = None, None, None, None

        for dds_path in dds_files:
            ref = self.parse_dds_bc1_endpoints(dds_path)
            W, H = ref["width"], ref["height"]
            Bx, By = ref["blocks_x"], ref["blocks_y"]

            if w0 is None:
                w0, h0, bx0, by0 = W, H, Bx, By
            else:
                if (W, H, Bx, By) != (w0, h0, bx0, by0):
                    raise ValueError(
                        f"Dimension mismatch in {dds_path.name}. "
                        f"Expected {w0}x{h0}, got {W}x{H}."
                    )
            
            refs.append(ref)
            texture_names.append(dds_path.stem)

        # Build bxby (N, 2)
        n = refs[0]["endpoints_rgb565"].shape[0]
        idx = np.arange(n, dtype=np.int32)
        bxby = np.stack([idx % bx0, idx // bx0], axis=-1)

        # Build concatenated Q01 endpoints
        ep_q01_list = []
        for ref in refs:
            eps = ref["endpoints_rgb565"]
            c0, c1 = eps[:, 0], eps[:, 1]
            q01_c0 = self._rgb565_to_q01_vec(c0)
            q01_c1 = self._rgb565_to_q01_vec(c1)
            ep_q01 = np.concatenate([q01_c0, q01_c1], axis=-1)  # (N, 6)
            ep_q01_list.append(ep_q01)

        ep_q01_all = np.concatenate(ep_q01_list, axis=-1)  # (N, 6*T)

        # Save Train_dataset.json
        dataset_path = self.output_dir / "Train_dataset.json"
        dataset_out = {
            "inputs": {"bxby": bxby.tolist()},
            "targets": {"ep_q01": ep_q01_all.tolist()},
        }
        
        if include_meta:
            dataset_out["meta"] = {
                "width": w0, "height": h0,
                "blocks_x": bx0, "blocks_y": by0,
                "num_textures": len(refs),
                "texture_names": texture_names
            }
            
        dataset_path.write_text(json.dumps(dataset_out))

        # Save Inference_input.json
        inference_path = self.output_dir / "Inference_input.json"
        inference_payload = {
            "blocks_x": bx0,
            "blocks_y": by0,
            "num_textures": len(refs),
            "texture_names": texture_names,
        }
        inference_path.write_text(json.dumps(inference_payload))

        print(f"Dataset created at: {dataset_path}")
        return str(dataset_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        extractor = EndpointExtractor(sys.argv[1], sys.argv[2])
        extractor.extract()
    else:
        print("Usage: python extract_endpoints.py <source_folder> <output_folder>")

