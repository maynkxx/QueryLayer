from rag_engine import build_vectorstore
from rag_engine import get_answer

# Build vector DB
vectorstore = build_vectorstore("Contest Absence and Marking Policy.pdf")

# Ask question
question = "When is absence allowed in contests?"

answer, docs = get_answer(question, vectorstore)

print("\n🤖 Answer:\n")
print(answer)

print("\n📄 Sources:\n")
for i, doc in enumerate(docs):
    print(f"\nSource {i+1}:")
    print(doc.page_content[:200])