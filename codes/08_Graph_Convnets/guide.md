# Graph Neural Networks Message Passing Guide

## Overview

This guide explains the message passing patterns used in Graph Convolutional Networks (GCNs), Graph Attention Networks (GATs), and Gated Graph Convolutional Networks (GatedGCNs) implemented with DGL.

All these networks follow the **message → reduce → update** paradigm:

1. **`message_func()`**: What messages to send between nodes
2. **`reduce_func()`**: How to aggregate messages at each node  
3. **`forward()`**: Overall coordination and final updates

---

## GCN (Graph Convolutional Network)

**Core Idea**: Degree-normalized aggregation of neighbor features

### Mathematical Formulation
```
h_i^(ℓ+1) = h_i^(ℓ) + ReLU(1/√d_i * Σ_{j∈N(i)} (1/√d_j) * W^(ℓ) * h_j^(ℓ))
```

### Implementation Pattern

```python
def message_func(self, edges):
    """Send: transformed neighbor features + neighbor degrees"""
    Whj = edges.src['Wh']  # W * h_j (transformed neighbor)
    dj = edges.src['d']    # degree of neighbor j
    return {'Whj': Whj, 'dj': dj}

def reduce_func(self, nodes):
    """Aggregate: degree-normalized sum of neighbor features"""
    Whj = nodes.mailbox['Whj']  # all neighbor features
    dj = nodes.mailbox['dj']    # all neighbor degrees
    inv_sqrt_dj = torch.pow(dj, -0.5)  # 1/√d_j
    h = torch.sum(inv_sqrt_dj * Whj, dim=1)  # normalized sum
    return {'h': h}

def forward(self, g, h):
    """Setup + final update"""
    h_in = h  # residual connection
    g.ndata['Wh'] = self.W(h)  # transform all nodes
    g.ndata['d'] = g.in_degrees().view(-1, 1).float()  # compute degrees
    g.update_all(self.message_func, self.reduce_func)  # message passing
    h_new = g.ndata['h']  # get aggregated features
    
    # Apply final transformations
    inv_sqrt_d = torch.pow(g.ndata['d'], -0.5)  # 1/√d_i
    h_new = inv_sqrt_d * h_new  # normalize by destination degree
    h_new = torch.relu(h_new)   # activation
    return h_in + h_new         # residual connection
```

**Key Features:**
- Symmetric normalization by source and destination degrees
- Simple uniform aggregation (no attention)
- Residual connections for training stability

---

## GAT (Graph Attention Network)

**Core Idea**: Attention-weighted aggregation of neighbor features

### Mathematical Formulation
```
h_i^(ℓ+1) = Concat_{k=1}^K (ELU(Σ_{j∈N_i} e_ij^(k,ℓ) * W_1^(k,ℓ) * h_j^ℓ))
e_ij^(k,ℓ) = Softmax(ê_ij^(k,ℓ))
ê_ij^(k,ℓ) = LeakyReLU(W_2^(k,ℓ) * Concat(W_1^(k,ℓ)*h_i^ℓ, W_1^(k,ℓ)*h_j^ℓ))
```

### Implementation Pattern

```python
def message_func(self, edges):
    """Send: transformed features + attention scores"""
    Whi = edges.dst['Wh']   # W * h_i (destination)
    Whj = edges.src['Wh']   # W * h_j (source)
    WhiWhj = torch.cat([Whi, Whj], dim=1)  # concatenate
    aWhiWhj = self.a(WhiWhj)  # attention network
    eij = nn.LeakyReLU()(aWhiWhj)  # raw attention score
    return {'eij': eij, 'Whj': Whj}

def reduce_func(self, nodes):
    """Aggregate: attention-weighted sum"""
    eij = nodes.mailbox['eij']   # raw attention scores
    Whj = nodes.mailbox['Whj']   # neighbor features
    exp_eij = torch.exp(eij)     # exponentiate for softmax
    # Softmax normalization + weighted sum
    h = torch.sum(exp_eij * Whj, dim=1) / torch.sum(exp_eij, dim=1)
    return {'h': h}

def forward(self, g, h):
    """Simple coordination"""
    g.ndata['Wh'] = self.W(h)  # transform features
    g.update_all(self.message_func, self.reduce_func)  # message passing
    h = g.ndata['h']           # get result
    return nn.ELU()(h)         # activation
```

**Multi-Head Attention:**
```python
class GAT_layer(nn.Module):
    def forward(self, g, h):
        list_h = []
        for gat_head in self.GAT_one_heads:
            h_head = gat_head(g, h)  # Each head processes independently
            list_h.append(h_head)
        h = torch.cat(list_h, dim=1)  # Concatenate all heads
        return h
```

**Key Features:**
- Learned attention weights determine neighbor importance
- Multi-head attention captures different relationship types
- Permutation invariant (order of neighbors doesn't matter)
- Self-attention mechanism similar to Transformers

---

## GatedGCN (Most Complex)

**Core Idea**: Edge features + gated aggregation with residual connections

### Mathematical Formulation
```
h_i^(ℓ+1) = h_i^ℓ + ReLU(BN(A^ℓ*h_i^ℓ + Σ_{j~i} η(e_ij^ℓ) ⊙ B^ℓ*h_j^ℓ))
e_ij^(ℓ+1) = e_ij^ℓ + ReLU(BN(C^ℓ*e_ij^ℓ + D^ℓ*h_i^ℓ + E^ℓ*h_j^ℓ))
η(e_ij^ℓ) = σ(e_ij^ℓ) / (Σ_{j'~i} σ(e_ij'^ℓ) + ε)
```

### Implementation Pattern

```python
def message_func(self, edges):
    """Send: neighbor features + edge updates"""
    Bhj = edges.src['Bh']  # B * h_j (neighbor features)
    # Edge update: C*e_ij + D*h_i + E*h_j
    eij = edges.data['Ce'] + edges.dst['Dh'] + edges.src['Eh']
    edges.data['e'] = eij  # ⚠️ Update edge features in-place
    return {'Bhj': Bhj, 'eij': eij}

def reduce_func(self, nodes):
    """Aggregate: gated sum (like attention but with sigmoid)"""
    Ahi = nodes.data['Ah']      # A * h_i (self features)
    Bhj = nodes.mailbox['Bhj']  # neighbor features
    e = nodes.mailbox['eij']    # edge features
    sigmaij = torch.sigmoid(e)  # gating weights
    h = Ahi + torch.sum(sigmaij * Bhj, dim=1)  # gated aggregation
    return {'h': h}

def forward(self, g, h, e, snorm_n, snorm_e):
    """Complex setup + dual updates"""
    h_in, e_in = h, e  # residual connections
    
    # Pre-compute all transformations (5 different matrices!)
    g.ndata['Ah'] = self.A(h)  # self-connection
    g.ndata['Bh'] = self.B(h)  # neighbor features
    g.ndata['Dh'] = self.D(h)  # for edge update (destination)
    g.ndata['Eh'] = self.E(h)  # for edge update (source)
    g.edata['Ce'] = self.C(e)  # edge self-connection
    
    g.update_all(self.message_func, self.reduce_func)  # message passing
    
    h = g.ndata['h'] * snorm_n  # normalize by graph size
    e = g.edata['e'] * snorm_e  # normalize by graph size
    
    # Apply batch norm + activation + residual
    h = self.bn_node_h(h)
    e = self.bn_node_e(e)
    h = torch.relu(h)
    e = torch.relu(e)
    h = h_in + h  # residual connection
    e = e_in + e  # residual connection
    
    return h, e  # ⚠️ Returns both node AND edge features
```

**Key Features:**
- Updates both node and edge features simultaneously
- Gating mechanism controls information flow
- Batch normalization for training stability
- Graph size normalization (snorm)
- Most parameters (5 linear transformations: A, B, C, D, E)

---

## Design Patterns & Common Components

### **What Goes Where?**

#### **`message_func()` contains:**
- **Feature transformation**: Apply linear layers to neighbor features
- **Attention/gating computation**: Calculate how much each neighbor matters
- **Edge updates** (GatedGCN only): Update edge features based on connected nodes
- **Pure computation**: No side effects (except edge updates in GatedGCN)

#### **`reduce_func()` contains:**
- **Aggregation strategy**: Sum, mean, max, or weighted combination
- **Normalization**: Softmax (GAT), degree normalization (GCN), sigmoid gating (GatedGCN)
- **Self vs neighbor combination**: How to combine node's own features with neighbors
- **Pure aggregation**: No neural network layers

#### **`forward()` contains:**
- **Pre-processing**: Transform input features, compute degrees/edges
- **Message passing coordination**: Call `g.update_all(message_func, reduce_func)`
- **Post-processing**: Batch norm, activation, residual connections
- **Neural network components**: All learnable layers go here

### **Common Confusion Points**

1. **Linear transformations**: Usually done in `forward()` before message passing, stored in node/edge data
2. **Attention computation**: Raw scores in `message_func()`, normalization in `reduce_func()`
3. **Residual connections**: Applied in `forward()` after message passing
4. **Batch normalization**: Always in `forward()`, never in message/reduce functions
5. **Edge updates**: Only GatedGCN updates edges; GCN and GAT focus on nodes only
6. **Activation functions**: Applied in `forward()` or as separate layers

### **Performance Considerations**

- **GCN**: Fastest, simplest computation
- **GAT**: Moderate speed, attention computation overhead
- **GatedGCN**: Slowest, most memory intensive (dual node/edge updates)

### **When to Use Which?**

- **GCN**: Simple tasks, homogeneous graphs, when interpretability matters
- **GAT**: Heterogeneous graphs, when you need to understand which neighbors matter
- **GatedGCN**: Complex tasks, when edge information is important, molecular graphs

---

## Molecular Property Prediction (Code05)

For molecular graphs, the key differences are:

1. **Node features**: Atom types (categorical) → embedding layers
2. **Edge features**: Bond types (categorical) → embedding layers  
3. **Task**: Graph-level regression (not node classification)
4. **Output**: Scalar prediction via MLP after graph pooling

```python
# Graph-level pooling
g.ndata['h'] = h
y = dgl.mean_nodes(g, 'h')  # Mean pooling over all nodes
y = self.MLP_layer(y)       # Two-layer MLP for regression
```

**Loss Function**: Mean Absolute Error (L1Loss) instead of CrossEntropy

```python
def loss(self, y_scores, y_labels):
    return nn.L1Loss()(y_scores, y_labels)  # MAE for regression
```

---

## Summary

The key insight is that **message_func** and **reduce_func** handle pure graph aggregation logic, while **forward()** manages the broader neural network concerns (transformations, normalization, activations). This separation makes the code modular and the message passing patterns reusable across different architectures.