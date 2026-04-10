from app.generation.rag_graph import build_graph

graph = build_graph()

while True:
    q = input("\nQuery: ")
    if q == "exit":
        break

    result = graph.invoke({"query": q})
    print("\nAnswer:\n", result["answer"])
