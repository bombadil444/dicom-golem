from yapapi.runner.ctx import WorkContext
import asyncio


async def worker(ctx: WorkContext, tasks):
    ctx.send_file(f"./data/dicom.tar.gz", "/golem/resource/dicom.tar.gz")
    async for task in tasks:
        partition = task.data
        ctx.begin()
        # TODO json currently not used
        ctx.send_json(
            "/golem/work/params.json",
            {"partition": partition, "column": 0, "step": 5000, "search": "203691"},
        )
        ctx.run("/golem/entrypoints/run")
        ctx.download_file(f"/golem/output/log.out", f"output/log.out")
        ctx.download_file(f"/golem/output/3d_clusters.png", f"output/3d_clusters.png")
        yield ctx.commit()
        # TODO: Check if job results are valid
        # and reject by: task.reject_task(msg = 'invalid file')
        task.accept_task()

    ctx.log("no more partitions to search")
