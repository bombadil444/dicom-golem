import numpy as np
from skimage import measure
from yapapi.runner import Engine, Task, vm
from datetime import timedelta
import asyncio
import math
import shutil
import logging
from time import time
from pathlib import Path
from worker import worker
from config import (
    MIN_MEM_GB,
    MIN_STORAGE_GB,
    MAX_WORKERS,
    BUDGET,
    SUBNET,
    OVERHEAD_MINUTES,
    PARTITION_SLICES,
    DATA_PATH,
    OUTPUT_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("main")


async def main():
    # get number of partitions
    data_path = Path(DATA_PATH)
    num_files = len(list(data_path.glob("*")))
    partitions = math.floor(num_files / PARTITION_SLICES)
    logger.info(f"{partitions=}")

    # calc timeout
    init_overhead: timedelta = timedelta(minutes=OVERHEAD_MINUTES)
    timeout = init_overhead + timedelta(minutes=partitions * 2)

    # get latest pushed package
    with open("hash_link", "r") as f:
        hash_link = f.read()

    package = await vm.repo(
        image_hash=hash_link.strip(),
        min_mem_gib=MIN_MEM_GB,
        min_storage_gib=MIN_STORAGE_GB,
    )

    Path("tmp/").mkdir()

    start_time = time()

    # start worker
    async with Engine(
        package=package,
        max_workers=MAX_WORKERS,
        budget=BUDGET,
        timeout=timeout,
        subnet_tag=SUBNET,
    ) as engine:
        async for progress in engine.map(
            worker, [Task(data=partition) for partition in range(0, partitions)]
        ):
            logger.info(f"{progress=}")

    logger.info(f"Execution time: {time() - start_time}")

    # create consolidated png
    output_path = Path(OUTPUT_PATH)
    clusters = []
    for np_file in sorted(output_path.rglob("*.npy")):
        cluster = np.load(str(np_file))
        clusters.append(cluster)

    all_clusters = np.concatenate(clusters)

    v, f = make_mesh(all_clusters, None)
    logger.info("Drawing 3D image")
    plt_3d(v, f, output_path / "3d_clusters")


def make_mesh(image, threshold=-300, step_size=1):
    p = image.transpose(2, 1, 0)
    verts, faces, norm, val = measure.marching_cubes(p, threshold)


def plt_3d(verts, faces, output_path):
    x, y, z = zip(*verts)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=1)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_zlim(0, 200)
    ax.set_facecolor((0.7, 0.7, 0.7))
    plt.savefig(output_path)


if __name__ == "__main__":
    # clear outputs from previous runs
    output_dir = Path(OUTPUT_PATH)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    loop = asyncio.get_event_loop()
    task = loop.create_task(main())
    try:
        asyncio.get_event_loop().run_until_complete(task)
    except (Exception, KeyboardInterrupt) as e:
        logger.error(e)
        task.cancel()
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.3))
    finally:
        if Path("tmp/").exists():
            shutil.rmtree(Path("tmp/"))
