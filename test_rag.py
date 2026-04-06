from rag_engine import build_vectorstore

# Build vector DB from your PDF
vectorstore = build_vectorstore("Contest Absence and Marking Policy.pdf")

# Create retriever
retriever = vectorstore.as_retriever()

# Ask a test question
# ✅ NEW
results = retriever.invoke("What is this policy about?")

# Print results
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content[:300])