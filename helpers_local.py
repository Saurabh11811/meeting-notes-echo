# helpers_local.py
from __future__ import annotations

def summarize_local_ollama(text: str, model: str = "llama3:latest", system_prompt: str = "") -> str:
    """
    Summarize using local Ollama via LangChain's Ollama wrapper.
    Requires: pip install langchain langchain_community
    """
    if not text or not text.strip():
        return "No transcript content to summarize."
    try:
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        from langchain_community.llms import Ollama
    except Exception as e:
        return f"[Local summarizer unavailable: {e}]"

    sys_prompt = system_prompt.strip() or (
        "Executive-ready summary. Provide Key Decisions, Action Items (owner & due date), Risks/Blockers, Notes."
    )

    llm = Ollama(model=model, temperature=0.1)
    tmpl = PromptTemplate(
        input_variables=["context"],
        template=(
            "<s><<SYS>>" + sys_prompt + "<</SYS>></s>\n"
            "[INST]\n{context}\n[/INST]"
        ),
    )
    chain = LLMChain(llm=llm, prompt=tmpl)
    return chain.run(context=text)
