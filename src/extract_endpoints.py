import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Tuple

DXGI_FORMAT_BC1_UNORM = 71
DXGI_FORMAT_BC1_UNORM_SRGB = 72


def rgb565_to_rgb888(c: int) -> Tuple[int, int, int]:
    r5 = (c >> 11) & 0x1F
    g6 = (c >> 5) & 0x3F
    b5 = c & 0x1F
    r = (r5 * 255 + 15) // 31
    g = (g6 * 255 + 31) // 63
    b = (b5 * 255 + 15) // 31
    return (int(r), int(g), int(b))


def rgb565_to_q01(c: int) -> List[float]:
    r5 = (c >> 11) & 0x1F
    g6 = (c >> 5) & 0x3F
    b5 = c & 0x1F
    return [r5 / 31.0, g6 / 63.0, b5 / 31.0]


def parse_dds_bc1_endpoints(dds_path: Path) -> Dict[str, Any]:
    data = dds_path.read_bytes()
    if len(data) < 128 or data[0:4] != b"DDS ":
        raise ValueError("Not a valid DDS file (missing DDS magic).")

    header = data[4 : 4 + 124]
    if struct.unpack_from("<I", header, 0)[0] != 124:
        raise ValueError("Unexpected DDS header size.")

    height = struct.unpack_from("<I", header, 8)[0]
    width = struct.unpack_from("<I", header, 12)[0]

    ddspf_off = 72
    fourcc = header[ddspf_off + 8 : ddspf_off + 12]

    offset = 4 + 124
    if fourcc == b"DXT1":
        pass
    elif fourcc == b"DX10":
        dx10 = data[offset : offset + 20]
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

    endpoints_rgb565: List[List[int]] = []
    endpoints_rgb888: List[List[List[int]]] = []

    p = offset
    for _ in range(num_blocks):
        c0 = int.from_bytes(data[p : p + 2], "little")
        c1 = int.from_bytes(data[p + 2 : p + 4], "little")
        endpoints_rgb565.append([c0, c1])
        endpoints_rgb888.append(
            [list(rgb565_to_rgb888(c0)), list(rgb565_to_rgb888(c1))]
        )
        p += 8

    return {
        "width": int(width),
        "height": int(height),
        "blocks_x": int(blocks_x),
        "blocks_y": int(blocks_y),
        "block_order": "row_major",
        "format": "BC1",
        "endpoints_rgb565": endpoints_rgb565,
        "endpoints_rgb888": endpoints_rgb888,
    }


def extract_endpoints_to_json(
    dds_filepath: str,
    output_folder: str = None,
    include_meta: bool = True,
    keep_only_c0_gt_c1: bool = False,
) -> str:
    """
    Takes a DDS filepath, extracts BC1 endpoints, converts them to the dataset format,
    and writes a JSON file to the output directory (or same as input) with a _endpoints.json suffix.
    Returns the path to the created JSON file.
    """
    dds_path = Path(dds_filepath).resolve()
    if not dds_path.exists():
        raise FileNotFoundError(f"File not found: {dds_path}")

    # Determine output path
    if output_folder:
        out_root = Path(output_folder).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
    else:
        out_root = dds_path.parent

    # 1. Parse DDS
    d = parse_dds_bc1_endpoints(dds_path)

    # 2. Convert to Dataset Format (Logic from convert_reference_to_dataset)
    W, H = int(d["width"]), int(d["height"])
    Bx, By = int(d["blocks_x"]), int(d["blocks_y"])
    eps = d["endpoints_rgb565"]
    n = len(eps)

    st: List[List[float]] = []
    bxby: List[List[int]] = []
    ep_q01: List[List[float]] = []
    c0_gt_c1: List[int] = []
    ep_rgb565: List[List[int]] = []

    for i, pair in enumerate(eps):
        c0, c1 = int(pair[0]), int(pair[1])
        bx = i % Bx
        by = i // Bx

        s = bx / (Bx - 1) if Bx > 1 else 0.0
        t = by / (By - 1) if By > 1 else 0.0

        flag = 1 if (c0 > c1) else 0
        if keep_only_c0_gt_c1 and flag == 0:
            continue

        bxby.append([bx, by])
        st.append([float(s), float(t)])

        ep_rgb565.append([c0, c1])
        ep_q01.append(rgb565_to_q01(c0) + rgb565_to_q01(c1))
        c0_gt_c1.append(flag)

    out: Dict[str, Any] = {
        "inputs": {"st": st, "bxby": bxby},
        "targets": {"ep_rgb565": ep_rgb565, "ep_q01": ep_q01},
        "flags": {"c0_gt_c1": c0_gt_c1},
    }

    if include_meta:
        out["meta"] = {
            "width": W,
            "height": H,
            "blocks_x": Bx,
            "blocks_y": By,
            "block_order": d.get("block_order", "row_major"),
            "format": d.get("format", "BC1"),
            "num_blocks_total": n,
            "num_blocks_kept": len(ep_rgb565),
            "filtered_c0_gt_c1": bool(keep_only_c0_gt_c1),
            "source_image": str(dds_path),
        }

    # Naming convention: [name]_endpoints.json
    out_json_path = out_root / f"{dds_path.stem}_endpoints.json"
    out_json_path.write_text(json.dumps(out, indent=2))
    print(f"Extracted endpoints to: {out_json_path}")
    return str(out_json_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        extract_endpoints_to_json(sys.argv[1])
    else:
        print("Usage: python extractendpoints.py <path_to_dds_file>")
