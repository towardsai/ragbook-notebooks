from llama_index.core import set_global_service_context
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import ServiceContext

# Set up the embedding model
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Set up the LLM
llm = OpenAI(model="gpt-4o-mini")

# Create a service context with these models
service_context = ServiceContext.from_defaults(
    embed_model=embed_model,
    llm=llm
)

# Set the global service context
set_global_service_context(service_context)

print("Global LlamaIndex settings have been configured.")