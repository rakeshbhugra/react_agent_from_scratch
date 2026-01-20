# Utility functions for LangGraph visualization

def draw_graph(graph, output_file: str = None, show_mermaid: bool = True):
    """
    Draw a LangGraph graph as mermaid diagram and optionally save as PNG.

    Args:
        graph: Compiled LangGraph graph
        output_file: Optional path to save PNG (e.g., "my_graph.png")
        show_mermaid: Print mermaid code to console (can paste at mermaid.live)

    Usage:
        from langgraph_tutorial.utils import draw_graph
        draw_graph(my_graph, "output.png")
    """
    # Get mermaid diagram text
    if show_mermaid:
        mermaid_code = graph.get_graph().draw_mermaid()
        print("Mermaid Diagram (paste at https://mermaid.live):")
        print("-" * 50)
        print(mermaid_code)
        print("-" * 50)

    # Save as PNG if output file specified
    if output_file:
        try:
            png_data = graph.get_graph().draw_mermaid_png()
            with open(output_file, "wb") as f:
                f.write(png_data)
            print(f"\nPNG saved to: {output_file}")
        except Exception as e:
            print(f"\nCouldn't save PNG: {e}")
            print("Try pasting the mermaid code at https://mermaid.live instead")


def draw_graph_ascii(graph):
    """
    Print a simple ASCII representation of graph nodes and edges.

    Args:
        graph: Compiled LangGraph graph
    """
    g = graph.get_graph()

    print("\nGraph Structure:")
    print("=" * 40)

    print("\nNodes:")
    for node in g.nodes:
        print(f"  - {node}")

    print("\nEdges:")
    for edge in g.edges:
        print(f"  {edge[0]} -> {edge[1]}")
