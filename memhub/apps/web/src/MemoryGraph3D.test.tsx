// @vitest-environment jsdom

import type {
  MemoryGraphNode,
  MemoryGraphResponse,
} from "@one-memory/contracts";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { MemoryGraph3D } from "./MemoryGraph3D";

afterEach(cleanup);

const NODE_TYPES: MemoryGraphNode["node_type"][] = [
  "memory",
  "source",
  "schema",
  "topic",
];

function makeGraph(nodeCount: number): MemoryGraphResponse {
  const nodes: MemoryGraphNode[] = Array.from(
    { length: nodeCount },
    (_, index) => ({
      node_id: `node-${index}`,
      node_type: NODE_TYPES[index % NODE_TYPES.length],
      label: `Node ${index}`,
      memory_id: null,
      source_id: null,
      kind: null,
      status: "active",
      confidence: 0.8,
    }),
  );
  return {
    total_memories: nodeCount,
    projected_memories: nodeCount,
    nodes,
    edges: nodes.slice(1).map((node, index) => ({
      edge_id: `edge-${index}`,
      source_node_id: "node-0",
      target_node_id: node.node_id,
      relation: "derived_from",
      weight: null,
      origin: "provenance" as const,
    })),
    hy_graph_available: true,
  };
}

describe("MemoryGraph3D", () => {
  it("gives an accessible fallback without WebGL", () => {
    render(
      <MemoryGraph3D
        graph={makeGraph(8)}
        selectedNodeId={null}
        onSelectNode={() => {}}
      />,
    );
    expect(
      screen.getByText(/This device cannot open the 3D view/),
    ).toBeDefined();
  });

  // Scene assembly (frame-split layout, entrance animation, on-demand rendering, selected state via ref instead of effect deps)
  // is not covered here, and that is not an oversight: jsdom has no WebGL, the effect throws when building the renderer and
  // returns early, so none of the assembly code runs. See OPEN-ITEMS A-31 for the restoration recipe.
  // Do not add a case here asserting the component was not unmounted: that one was tested, and it is green with or without the fix.
});
