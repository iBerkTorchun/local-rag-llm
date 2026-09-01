from foundry_local_sdk import Configuration, FoundryLocalManager


APP_NAME = "foundry_local_rag"
CUDA_EXECUTION_PROVIDER = "CUDAExecutionProvider"


def initialize_foundry_manager() -> FoundryLocalManager:
    """Initialize Foundry Local and make supported CUDA execution available."""
    FoundryLocalManager.initialize(Configuration(app_name=APP_NAME))
    manager = FoundryLocalManager.instance

    try:
        result = manager.download_and_register_eps(
            names=[CUDA_EXECUTION_PROVIDER]
        )
        cuda_registered = any(
            provider.name == CUDA_EXECUTION_PROVIDER and provider.is_registered
            for provider in manager.discover_eps()
        )
        if result.success and not result.failed_eps and cuda_registered:
            print("[OK] CUDA execution provider available.")
        else:
            print(
                "[WARN] CUDA execution provider unavailable; continuing with "
                f"automatic alias resolution. {result.status}"
            )
    except Exception as error:
        print(
            "[WARN] CUDA execution provider registration failed; continuing "
            f"with automatic alias resolution. {error}"
        )

    # Successful EP registration invalidates the SDK catalog. Refresh it before
    # resolving aliases so Foundry Local can rank the newly available variants.
    manager.catalog.list_models()
    return manager
