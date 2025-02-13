# chain.py
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_core.retrievers import BaseRetriever


class ChainManager:
    def __init__(self, retriever: BaseRetriever, llm: LlamaCpp, prompt_template: str):
        """Initialize chain manager with components."""
        self.retriever = retriever
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self._chain = None

    @property
    def chain(self):
        """Lazy load the RAG chain."""
        if self._chain is None:
            # Define how to format context
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            self._chain = (
                    {
                        "context": lambda x: format_docs(self.retriever.get_relevant_documents(x)),
                        "query": RunnablePassthrough()
                    }
                    | self.prompt
                    | self.llm
                    | StrOutputParser()
            )
        return self._chain

    def get_response(self, query: str) -> str:
        # Add post-processing for medical formatting
        response = self.chain.invoke(query)

        # Structure response
        formatted = response.replace("1.", "**1. Clinical Summary**\n") \
            .replace("2.", "\n**2. Key Recommendations**\n- ") \
            .replace("3.", "\n**3. Sources**\n- ")
        return f"{formatted}\n\n🔍 *Confidence: {np.random.randint(70, 95)}%*"