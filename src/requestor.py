from yapapi.runner import Engine, Task, vm
from datetime import timedelta
import asyncio
import os
from worker import worker
from config import (
    MIN_MEM_GB,
    MIN_STORAGE_GB,
    MAX_WORKERS,
    BUDGET,
    SUBNET,
    OVERHEAD_MINUTES,
    NUM_WORKERS,
)


async def main():
    # TODO make this dynamic. Iterator over the indicies in the partitions
    partitions: range = range(0, NUM_WORKERS)

    # TODO make this dynamic, e.g. depending on the size of files to transfer
    # worst-case time overhead for initialization, e.g. negotiation, file transfer etc.
    init_overhead: timedelta = timedelta(minutes=OVERHEAD_MINUTES)

    with open("hash_link", "r") as f:
        hash_link = f.read()

    package = await vm.repo(
        image_hash=hash_link.strip(),
        min_mem_gib=MIN_MEM_GB,
        min_storage_gib=MIN_STORAGE_GB,
    )

    async with Engine(
        package=package,
        max_workers=MAX_WORKERS,
        budget=BUDGET,
        timeout=init_overhead + timedelta(minutes=len(partitions) * 2),
        subnet_tag=SUBNET,
    ) as engine:
        async for progress in engine.map(
            worker, [Task(data=partition) for partition in partitions]
        ):
            print("progress=", progress)


if __name__ == "__main__":
    # clear outputs from previous runs
    folder = "output/"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)

    loop = asyncio.get_event_loop()
    task = loop.create_task(main())
    try:
        asyncio.get_event_loop().run_until_complete(task)
    except (Exception, KeyboardInterrupt) as e:
        print(e)
        task.cancel()
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.3))
