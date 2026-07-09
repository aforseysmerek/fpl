"""
Visualize collected episodes so you can calibrate the reward signals: writes a
per-demo mp4 of the agent_view *named with its axis scores*, plus a printed +
CSV table of every episode's scores. Works on any episodes.hdf5 (square or wipe).

The point: watch a demo, then check whether its scores match what you see
(e.g. does a visibly circular wipe get a high `circularity`? a hard press a high
`pressing`?). That tells us how to tune the reward functions.

Run in robodiff:
    python scripts/visualize_episodes.py --episodes shared_data_wipe/episodes.hdf5 -o viz_wipe
"""
import os
import csv
import pathlib
import click
import numpy as np
import h5py


def numeric_attrs(demo):
    out = {}
    for k, v in demo.attrs.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            pass
    return out


@click.command()
@click.option('--episodes', required=True)
@click.option('-o', '--out_dir', default=None)
@click.option('--fps', type=int, default=20)
@click.option('--max_videos', type=int, default=20)
def main(episodes, out_dir, fps, max_videos):
    out_dir = out_dir or str(pathlib.Path(episodes).parent / "viz")
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    f = h5py.File(episodes, "r")
    demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
    rows = []
    for i, k in enumerate(demos):
        d = f["data"][k]
        attrs = numeric_attrs(d)
        rows.append({"demo": k, **attrs})
        if i < max_videos and "agent_view" in d["obs"]:
            frames = d["obs"]["agent_view"][:]              # (T,H,W,3) uint8
            tag = "_".join(f"{n}{attrs[n]:.2f}" for n in list(attrs)[:3])
            path = os.path.join(out_dir, f"{k}__{tag}.mp4")
            try:
                import imageio
                imageio.mimwrite(path, list(frames), fps=fps)
            except Exception as e:
                try:  # fallback: 8-frame contact sheet
                    from PIL import Image
                    idx = np.linspace(0, len(frames) - 1, 8).astype(int)
                    strip = np.concatenate([frames[j] for j in idx], axis=1)
                    Image.fromarray(strip).save(path.replace(".mp4", ".png"))
                except Exception as e2:
                    print(f"  ({k}) viz skipped: {e} / {e2}")
    f.close()

    keys = sorted({k for r in rows for k in r if k != "demo"})
    print("\n=== per-episode scores ===")
    print("demo".ljust(10) + "".join(k.ljust(16) for k in keys))
    for r in rows:
        print(r["demo"].ljust(10) + "".join(f"{r.get(k, float('nan')):<16.3f}" for k in keys))
    with open(os.path.join(out_dir, "scores.csv"), "w", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=["demo"] + keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nVideos + scores.csv in {out_dir}/ — watch the mp4s (filenames carry their scores) "
          f"and check the numbers match what you see.")


if __name__ == '__main__':
    main()
