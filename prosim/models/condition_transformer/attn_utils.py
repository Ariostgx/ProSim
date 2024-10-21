import torch

def obtain_valid_edge_node_idex(edge_mask, node_mask):
  '''
    Obtain global valid edge indices from edge mask and node mask

    Global edge indices: 
      
      flattened node indices in a batch [0, P), where P is the number of valid nodes in the batch
      P = node_mask.sum()

      Usage:
        We can directly use the these indices to index the valid nodes in the batch:
          start_nodes = node_emds[node_mask][edge_node_idex[:, 0]]
          end_nodes = node_emds[node_mask][edge_node_idex[:, 1]]


    Input:
      edge_mask (tensor): [B, N, N] - binary edge mask
      node_mask (tensor): [B, N] - binary node mask

    Output:
      edge_node_idex (tensor): [2, E] - valid edge indices in global node indices
        E = edge_mask.sum() - number of valid edges
  '''
  B, N = node_mask.shape

  device = edge_mask.device

  # Global node indices
  node_indices = torch.ones(B, N, dtype=torch.long, device=device) * -1  # Size: [B, N]
  flat_node_indices = torch.arange(node_mask.sum(), device=device)  # Re-index to [0, P)
  node_indices[node_mask] = flat_node_indices  # Update global indices, # Size: [B, N]

  # Flatten edges and filter valid ones
  valid_edges = edge_mask.nonzero()  # Size: [E, 3] (E is number of valid edges)
  valid_edges = valid_edges[edge_mask[valid_edges[:, 0], valid_edges[:, 1], valid_edges[:, 2]]]

  # Create a map from 2D node indices to flattened 1D indices
  node_map = -torch.ones(B, N, dtype=torch.long, device=device)  # Initialize with -1 (invalid)
  node_map[node_mask] = torch.arange(flat_node_indices.size(0), device=device)  # Map to valid indices

  # Map to 1D indices
  start = flat_node_indices[node_map[valid_edges[:, 0], valid_edges[:, 1]]]  # Size: [E]
  end = flat_node_indices[node_map[valid_edges[:, 0], valid_edges[:, 2]]]  # Size: [E]
  edge_node_index = torch.stack([start, end], dim=0)  # Size: [2, E]

  return edge_node_index

if __name__ == '__main__':
  # Settings
  B, N = 2, 4  # Batches = 2, Nodes per batch = 4
  # Edge matrix (randomly generated for demo)
  edge_mask = torch.rand(B, N, N) > 0.5  # Size: [B, N, N]
  # Valid mask (randomly generated for demo)
  node_mask = torch.rand(B, N) > 0.3  # Size: [B, N]

  print('Edge mask:' + str(edge_mask))
  print('Node mask:' + str(node_mask))
  print('Valid edge indices:' + str(obtain_valid_edge_node_idex(edge_mask, node_mask)))