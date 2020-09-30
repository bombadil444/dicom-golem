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

    # TODO create consolidated png


if __name__ == "__main__":
    # clear outputs from previous runs
    output_dir = Path("output/")
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
