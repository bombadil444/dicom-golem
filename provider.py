from yapapi.runner import Engine, Task, vm
from yapapi.runner.ctx import WorkContext
from datetime import timedelta
import asyncio
import os


async def main():
    folder = 'output/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)

    with open('hash_link','r') as f:
        hash_link = f.read()

    package = await vm.repo(
        image_hash=hash_link.strip(),
        min_mem_gib=0.5,
        min_storage_gib=2.0,
    )

    async def worker(ctx: WorkContext, tasks):
        ctx.send_file(f"./data/example.csv", "/golem/resource/example.csv")
        async for task in tasks:
            partition = task.data
            ctx.begin()
            ctx.send_json(
                "/golem/work/params.json",
                {
                    "partition": partition,
                    "column": 0,
                    "step": 5000,
                    "search": "203691",
                },
            )
            ctx.run("/golem/entrypoints/run.sh")
            ctx.download_file(f"/golem/output/output_{partition}.txt", f"output/{partition}.txt")
            yield ctx.commit()
            # TODO: Check if job results are valid
            # and reject by: task.reject_task(msg = 'invalid file')
            task.accept_task()

        ctx.log("no more partitions to search")

    # iterator over the indicies in the partitions
    partitions: range = range(0, 7)
    # TODO make this dynamic, e.g. depending on the size of files to transfer
    # worst-case time overhead for initialization, e.g. negotiation, file transfer etc.
    init_overhead: timedelta = timedelta(minutes=3)

    async with Engine(
        package=package,
        max_workers=2,
        budget=10.0,
        timeout=init_overhead + timedelta(minutes=len(partitions) * 2),
        subnet_tag="testnet",
    ) as engine:

        async for progress in engine.map(worker, [Task(data=partition) for partition in partitions]):
            print("progress=", progress)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    task = loop.create_task(main())
    try:
        asyncio.get_event_loop().run_until_complete(task)
    except (Exception, KeyboardInterrupt) as e:
        print(e)
        task.cancel()
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.3))
