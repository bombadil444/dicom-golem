from yapapi.runner.ctx import WorkContext
import asyncio


async def worker(ctx: WorkContext, tasks):
    ctx.send_file(f"./data/C4.tar.gz", "/golem/resource/C4.tar.gz")
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
        ctx.run("/golem/entrypoints/run")
        ctx.download_file(f"/golem/output/log.out", f"output/log.out")
        ctx.download_file(f"/golem/output/python.log", f"output/python.log")
        ctx.download_file(f"/golem/output/test2.png", f"output/test2.png")
        yield ctx.commit()
        # TODO: Check if job results are valid
        # and reject by: task.reject_task(msg = 'invalid file')
        task.accept_task()

    ctx.log("no more partitions to search")
