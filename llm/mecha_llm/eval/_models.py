"""Judge models for the evaluators (RAGAS + DeepEval).

The judges run on the same deployment as the model under test
(``azure_openai_chat_deployment``). Note the consequence: swapping that
deployment moves the measuring stick as well as the thing being measured, so
scores are only comparable against runs on the same model.

Judges grade at temperature 0 for deterministic scoring, but drop the parameter
when the deployment doesn't accept one (see ``chat_temperature``).

Imports are lazy (inside the functions) so this module — and the modules that
import it — stay importable without the optional ``eval`` dependencies installed
(e.g. for pytest collection).
"""

from ..config import settings


def _judge_temperature() -> dict:
    """``{"temperature": 0}``, or nothing if the model rejects the parameter."""
    return {} if settings.chat_temperature is None else {"temperature": 0}


def ragas_llm():
    from langchain_openai import AzureChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_chat_deployment,
            **_judge_temperature(),
        )
    )


def ragas_embeddings():
    from langchain_openai import AzureOpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(
        AzureOpenAIEmbeddings(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_embedding_deployment,
        )
    )


def deepeval_model():
    from deepeval.models import AzureOpenAIModel

    return AzureOpenAIModel(
        model=settings.azure_openai_chat_deployment,
        deployment_name=settings.azure_openai_chat_deployment,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        base_url=settings.azure_openai_endpoint,
        **_judge_temperature(),
    )
