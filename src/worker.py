from yapapi.runner.ctx import WorkContext
import asyncio
import tarfile
import math
from pathlib import Path
from config import PARTITION_SLICES, DATA_PATH


async def worker(ctx: WorkContext, tasks):
    data_path = Path(DATA_PATH)
    num_files = len(list(data_path.glob("*")))

    async for task in tasks:
        partition = task.data

        start_index = partition * PARTITION_SLICES

        # if it's the final partition, iterate all the way to the end
        if partition == math.floor(num_files / PARTITION_SLICES):
            end_index = num_files
        else:
            end_index = start_index + PARTITION_SLICES

        tar_path = f"tmp/dicom_{partition}.tar.gz"

        with tarfile.open(tar_path, mode="w:gz") as dicom_tar:
            for i, dicom_path in enumerate(data_path.rglob("*.dcm")):
                if start_index <= i < end_index:
                    dicom_tar.add(str(dicom_path))

        ctx.send_file(tar_path, "/golem/resource/dicom.tar.gz")

        ctx.begin()
        # TODO send config info to container
        ctx.run("/golem/entrypoints/run")
        ctx.download_file(f"/golem/output/log.out", f"output/log_{partition}.out")

        # TODO download .npy file instead of .png
        ctx.download_file(
            f"/golem/output/3d_clusters.png", f"output/3d_clusters_{partition}.png"
        )
        yield ctx.commit()

        # TODO: Check if job results are valid
        # and reject by: task.reject_task(msg = 'invalid file')
        task.accept_task()

    ctx.log("no more partitions to search")
