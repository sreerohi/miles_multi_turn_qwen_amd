import torch
from torch import multiprocessing as mp
from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync


def _patch_filesystem_async_fork():
    """Replace the fork context with spawn in FileSystemWriterAsync.write_preloaded_data_multiproc.

    mp.get_context("fork") crashes on ROCm: forked children inherit the parent's
    HIP context and segfault inside torch.save. Subclass override doesn't work
    because the method is called internally via the original class reference.
    We patch the bound method on the class directly.
    """
    import megatron.core.dist_checkpointing.strategies.filesystem_async as fa_mod

    original = fa_mod.FileSystemWriterAsync.write_preloaded_data_multiproc

    def _patched_write_preloaded_data_multiproc(*args, **kwargs):
        # Intercept by temporarily swapping mp.get_context inside the module
        original_get_context = fa_mod.mp.get_context

        def _spawn_context(method):
            if method == "fork":
                print("[ROCm] write_preloaded_data_multiproc: redirecting fork -> spawn")
                return original_get_context("spawn")
            return original_get_context(method)

        fa_mod.mp.get_context = _spawn_context
        try:
            return original(*args, **kwargs)
        finally:
            fa_mod.mp.get_context = original_get_context

    fa_mod.FileSystemWriterAsync.write_preloaded_data_multiproc = staticmethod(
        _patched_write_preloaded_data_multiproc
    )
    print("[ROCm] filesystem_async: patched write_preloaded_data_multiproc fork -> spawn")


class ROCmFileSystemWriterAsync(FileSystemWriterAsync):
    """
    FileSystemWriterAsync wrapper for ROCm compatibility.

    On ROCm/HIP, using non_blocking=True causes tensors to be stored in pinned memory,
    which triggers segmentation faults when forking subprocesses afterward.
    """

    @staticmethod
    def preload_tensors(*args, **kwargs):
        if torch.version.hip:
            print("HIP/ROCm detected: setting non_blocking=False in preload_tensors")
            if "non_blocking" in kwargs:
                kwargs["non_blocking"] = False
            elif len(args) > 1 and isinstance(args[-1], bool):
                args = args[:-1] + (False,)

        return FileSystemWriterAsync.preload_tensors(*args, **kwargs)


if torch.version.hip:
    _patch_filesystem_async_fork()
