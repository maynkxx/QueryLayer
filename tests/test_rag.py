from rag_engine import build_vectorstore

vectorstore = build_vectorstore("Contest Absence and Marking Policy.pdf")

retriever = vectorstore.as_retriever()


results = retriever.invoke("What is this policy about?")

for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content[:300])