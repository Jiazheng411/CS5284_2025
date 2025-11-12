# Complete Graph Neural Networks Implementation Guide

## Overview

This comprehensive guide covers the implementation patterns used in both **Graph Convolutional Networks** (08_Graph_Convnets) and **Graph Transformers** (10_Graph_Transformers). All architectures follow the fundamental **message → reduce → update** paradigm but with different levels of complexity.

**Covered Architectures:**
- **GCN, GAT, GatedGCN** (08_Graph_Convnets)
- **Vanilla Graph Transformers, Graph Transformers with Positional Encoding, Graph Transformers with Edge Features** (10_Graph_Transformers)

---

## Essential DGL Operations & Common Patterns

### **Basic DGL Graph Operations**

```python
# Calculate in-degrees for normalization
g.in_degrees()              # Returns tensor of in-degrees for each node
g.in_degrees().float()      # Convert to float for calculations
g.in_degrees().view(-1, 1)  # Reshape for broadcasting [N, 1]

# Calculate out-degrees
g.out_degrees()             # Returns tensor of out-degrees for each node

# Graph properties
g.number_of_nodes()         # Total number of nodes
g.number_of_edges()         # Total number of edges

# Node and edge data storage
g.ndata['key'] = tensor     # Store node features
g.edata['key'] = tensor     # Store edge features
```

### **Batch Creation (Collate Functions)**

```python
def collate(samples):
    """Standard collate function for graph batching"""
    # Input: list of (graph, label) pairs
    graphs, labels = map(list, zip(*samples))
    
    # Batch graphs into single large graph
    batched_graphs = dgl.batch(graphs)
    
    # Stack labels into tensor
    batched_labels = torch.stack(labels)
    
    return batched_graphs, batched_labels

# Usage with DataLoader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate)
```

### **Positional Encoding (Graph Transformers)**

```python
def LapEig_positional_encoding(g, pos_enc_dim):
    """Compute Laplacian eigenvector positional encoding"""
    # Get adjacency matrix
    Adj = g.adj().to_dense()
    
    # Compute degree normalization matrix
    Dn = (g.in_degrees() ** -0.5).diag()
    
    # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    Lap = torch.eye(g.number_of_nodes()) - Dn.matmul(Adj).matmul(Dn)
    
    # Eigendecomposition
    EigVal, EigVec = torch.linalg.eig(Lap)
    EigVal, EigVec = EigVal.real, EigVec.real
    
    # Sort by eigenvalues and select non-trivial eigenvectors
    EigVec = EigVec[:, EigVal.argsort()]
    EigVec = EigVec[:, 1:pos_enc_dim+1]  # Skip first (trivial) eigenvector
    
    return EigVec
```

### **Common YOUR CODE Patterns**

```python
# Typical embedding initialization
self.embedding_h = nn.Embedding(num_node_types, hidden_dim)
self.embedding_e = nn.Embedding(num_edge_types, hidden_dim)

# Linear transformations for multi-head attention
g.ndata['Q'] = Q.view(-1, num_heads, head_hidden_dim)
g.ndata['K'] = K.view(-1, num_heads, head_hidden_dim)
g.ndata['V'] = V.view(-1, num_heads, head_hidden_dim)

# Standard message passing call
g.update_all(self.message_func, self.reduce_func)

# Graph-level pooling for graph classification/regression
g.ndata['h'] = h
graph_representation = dgl.mean_nodes(g, 'h')  # or dgl.sum_nodes, dgl.max_nodes
```

---

## Graph Convolutional Networks (08_Graph_Convnets)

### **GCN (Graph Convolutional Network)**

**Core Idea**: Degree-normalized aggregation of neighbor features

#### Mathematical Formulation
```
h_i^(ℓ+1) = h_i^(ℓ) + ReLU(1/√d_i * Σ_{j∈N(i)} (1/√d_j) * W^(ℓ) * h_j^(ℓ))
```

#### Implementation Pattern

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

### **GAT (Graph Attention Network)**

**Core Idea**: Attention-weighted aggregation of neighbor features

#### Mathematical Formulation
```
h_i^(ℓ+1) = Concat_{k=1}^K (ELU(Σ_{j∈N_i} e_ij^(k,ℓ) * W_1^(k,ℓ) * h_j^ℓ))
e_ij^(k,ℓ) = Softmax(ê_ij^(k,ℓ))
ê_ij^(k,ℓ) = LeakyReLU(W_2^(k,ℓ) * Concat(W_1^(k,ℓ)*h_i^ℓ, W_1^(k,ℓ)*h_j^ℓ))
```

#### Implementation Pattern

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

### **GatedGCN (Most Complex)**

**Core Idea**: Edge features + gated aggregation with residual connections

#### Mathematical Formulation
```
h_i^(ℓ+1) = h_i^ℓ + ReLU(BN(A^ℓ*h_i^ℓ + Σ_{j~i} η(e_ij^ℓ) ⊙ B^ℓ*h_j^ℓ))
e_ij^(ℓ+1) = e_ij^ℓ + ReLU(BN(C^ℓ*e_ij^ℓ + D^ℓ*h_i^ℓ + E^ℓ*h_j^ℓ))
η(e_ij^ℓ) = σ(e_ij^ℓ) / (Σ_{j'~i} σ(e_ij'^ℓ) + ε)
```

#### Implementation Pattern

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
```

---

## Graph Transformers (10_Graph_Transformers)

### **Vanilla Graph Transformers (Code01)**

**Core Idea**: Self-attention on graph structure without positional encoding

#### Mathematical Formulation
```
h_i^(ℓ+1) = h_i^ℓ + gMHA(LN(h^ℓ))
gHA(h)_i = Σ_{j∈N_i} Softmax(q_i^T k_j / √d') * v_j
```

#### Implementation Pattern

```python
def message_func(self, edges):
    """Send: Q-K dot products and V values"""
    qi = edges.dst['Q']  # queries from destination nodes
    kj = edges.src['K']  # keys from source nodes
    vj = edges.src['V']  # values from source nodes
    
    # Compute attention scores: q_i^T * k_j
    qikj = (qi * kj).sum(dim=2).unsqueeze(2)  # [E, num_heads, 1]
    
    # Exponentiate for softmax (normalized in reduce_func)
    expij = torch.exp(qikj / torch.sqrt(torch.tensor(self.head_hidden_dim)))
    
    return {'expij': expij, 'vj': vj}

def reduce_func(self, nodes):
    """Aggregate: softmax attention over neighbors"""
    expij = nodes.mailbox['expij']  # [N, |N_i|, num_heads, 1]
    vj = nodes.mailbox['vj']        # [N, |N_i|, num_heads, d']
    
    # Softmax normalization
    numerator = (expij * vj).sum(dim=1)    # weighted sum of values
    denominator = expij.sum(dim=1)         # sum of attention weights
    
    h = numerator / denominator  # softmax attention
    return {'h': h}

def forward(self, g, h):
    """Standard transformer setup"""
    # Compute Q, K, V transformations
    Q = self.WQ(h)  # [N, d]
    K = self.WK(h)  # [N, d]
    V = self.WV(h)  # [N, d]
    
    # Reshape for multi-head attention
    g.ndata['Q'] = Q.view(-1, self.num_heads, self.head_hidden_dim)
    g.ndata['K'] = K.view(-1, self.num_heads, self.head_hidden_dim)
    g.ndata['V'] = V.view(-1, self.num_heads, self.head_hidden_dim)
    
    # Perform message passing
    g.update_all(self.message_func, self.reduce_func)
    
    gMHA = g.ndata['h']  # [N, num_heads, head_hidden_dim]
    return gMHA
```

### **Graph Transformers with Positional Encoding (Code02)**

**Enhancement**: Adds Laplacian eigenvector positional encoding

#### Additional Components

```python
def forward(self, g, h, pe, e):
    """Include positional encoding in input"""
    # Node embedding with positional encoding
    h = self.embedding_h(h)           # atom/node type embedding
    h = h + self.embedding_pe(pe)     # add positional encoding
    
    # Rest is same as vanilla GT...
```

**Key Addition**: Positional encoding helps the model understand graph structure beyond just connectivity.

### **Graph Transformers with Edge Features (Code03 - DGL Sparse)**

**Enhancement**: Incorporates edge features into attention computation

#### Mathematical Formulation
```
gHA(h,e)_i = Σ_{j∈N_i} Softmax(q_i^T diag(e_ij) k_j / √d') * v_j
gHE(e,h)_ij = q_i ⊙ e_ij ⊙ k_j / √d'
```

#### Implementation Pattern

```python
def message_func(self, edges):
    """Send: edge-modulated attention + edge updates"""
    qi = edges.dst['Q']  # destination queries
    kj = edges.src['K']  # source keys
    vj = edges.src['V']  # source values
    eij = edges.data['E']  # edge features
    
    # Edge-modulated attention: q_i^T * diag(e_ij) * k_j
    qikj = (qi * eij * kj).sum(dim=2).unsqueeze(2)  # element-wise product
    expij = torch.exp(qikj / torch.sqrt(torch.tensor(self.head_hidden_dim)))
    
    # Edge feature update: q_i ⊙ e_ij ⊙ k_j
    fi = edges.dst['F']  # additional node features for edge update
    gj = edges.src['G']  # additional node features for edge update
    edge_update = fi * eij * gj / torch.sqrt(torch.tensor(self.head_hidden_dim))
    edges.data['e'] = edge_update  # update edge features
    
    return {'expij': expij, 'vj': vj}

def reduce_func(self, nodes):
    """Same softmax attention as vanilla GT"""
    expij = nodes.mailbox['expij']
    vj = nodes.mailbox['vj']
    
    numerator = (expij * vj).sum(dim=1)
    denominator = expij.sum(dim=1)
    h = numerator / denominator
    
    return {'h': h}

def forward(self, g, h, e):
    """Setup with both node and edge features"""
    # Transform node features
    Q = self.WQ(h)
    K = self.WK(h)
    V = self.WV(h)
    F = self.WF(h)  # for edge updates
    G = self.WG(h)  # for edge updates
    
    # Transform edge features
    E = self.WE(e)
    
    # Store in graph
    g.ndata['Q'] = Q.view(-1, self.num_heads, self.head_hidden_dim)
    g.ndata['K'] = K.view(-1, self.num_heads, self.head_hidden_dim)
    g.ndata['V'] = V.view(-1, self.num_heads, self.head_hidden_dim)
    g.ndata['F'] = F.view(-1, self.num_heads, self.head_hidden_dim)
    g.ndata['G'] = G.view(-1, self.num_heads, self.head_hidden_dim)
    g.edata['E'] = E.view(-1, self.num_heads, self.head_hidden_dim)
    
    g.update_all(self.message_func, self.reduce_func)
    
    gMHA = g.ndata['h']  # node updates
    gMHE = g.edata['e']  # edge updates
    return gMHA, gMHE
```

### **Dense Graph Transformers (Code04 - PyTorch Dense)**

**Key Difference**: Uses dense adjacency matrices instead of sparse DGL operations

#### Implementation Pattern

```python
class head_attention(nn.Module):
    def forward(self, x, e):
        """Dense attention computation"""
        Q = self.Q(x)  # [batch, n, d_head]
        K = self.K(x)  # [batch, n, d_head]
        V = self.V(x)  # [batch, n, d_head]
        
        # Expand for all-pairs computation
        Q = Q.unsqueeze(2)  # [batch, n, 1, d_head]
        K = K.unsqueeze(1)  # [batch, 1, n, d_head]
        
        # Edge features processing
        E = self.E(e)  # [batch, n, n, d_head]
        Ni = self.Ni(x).unsqueeze(2)  # [batch, n, 1, d_head]
        Nj = self.Nj(x).unsqueeze(1)  # [batch, 1, n, d_head]
        
        # Combine edge and node information
        e = Ni + Nj + E
        
        # Dense attention: Q * e * K for all pairs
        Att = (Q * e * K).sum(dim=3) / self.sqrt_d  # [batch, n, n]
        Att = torch.softmax(Att, dim=1)  # softmax over source nodes
        
        # Apply attention to values
        x = Att @ V  # [batch, n, d_head]
        return x, e
```

---

## Layer Components & Architecture Patterns

### **GraphTransformer Layer Structure**

```python
class GraphTransformer_layer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        # Multi-head attention
        self.gMHA = graph_MHA_layer(hidden_dim, hidden_dim//num_heads, num_heads)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # MLP block
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection and dropout
        self.WO = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_mha = nn.Dropout(dropout)
        self.dropout_mlp = nn.Dropout(dropout)

    def forward(self, g, h, e=None):
        """Standard transformer layer: MHA + residual + MLP + residual"""
        # Multi-head attention block
        h_rc = h  # residual connection
        h = self.layer_norm1(h)  # pre-norm
        
        if e is not None:
            h_MHA, e_MHA = self.gMHA(g, h, e)  # with edge features
        else:
            h_MHA = self.gMHA(g, h)  # vanilla
            
        h_MHA = h_MHA.view(-1, self.hidden_dim)  # flatten heads
        h_MHA = self.dropout_mha(h_MHA)
        h_MHA = self.WO(h_MHA)
        h = h_rc + h_MHA  # residual connection
        
        # MLP block
        h_rc = h
        h = self.layer_norm2(h)  # pre-norm
        h_MLP = self.linear1(h)
        h_MLP = torch.relu(h_MLP)
        h_MLP = self.dropout_mlp(h_MLP)
        h_MLP = self.linear2(h_MLP)
        h = h_rc + h_MLP  # residual connection
        
        if e is not None:
            return h, e_MHA
        else:
            return h
```

### **Full Network Architecture**

```python
class GraphTransformer_net(nn.Module):
    def __init__(self, net_parameters):
        super().__init__()
        # Input embeddings
        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)
        self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)  # if edge features
        self.embedding_pe = nn.Linear(pos_enc_dim, hidden_dim)      # if positional encoding
        
        # Transformer layers
        self.GraphTransformer_layers = nn.ModuleList([
            GraphTransformer_layer(hidden_dim, num_heads) 
            for _ in range(L)
        ])
        
        # Output layers
        self.ln_final = nn.LayerNorm(hidden_dim)
        self.linear_final = nn.Linear(hidden_dim, num_classes)

    def forward(self, g, h, pe=None, e=None):
        """Full forward pass"""
        # Input embeddings
        h = self.embedding_h(h)
        
        if pe is not None:
            h = h + self.embedding_pe(pe)  # add positional encoding
            
        if e is not None:
            e = self.embedding_e(e)        # edge embeddings
        
        # Transformer layers
        for GT_layer in self.GraphTransformer_layers:
            if e is not None:
                h, e = GT_layer(g, h, e)   # with edge features
            else:
                h = GT_layer(g, h)         # vanilla
        
        # Output processing
        if self.task == 'graph_classification':
            g.ndata['h'] = h
            graph_repr = dgl.mean_nodes(g, 'h')  # graph-level pooling
            y = self.ln_final(graph_repr)
            y = self.linear_final(y)
        else:  # node classification
            y = self.ln_final(h)
            y = self.linear_final(y)
            
        return y
```

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

---

## Common Implementation Patterns

### **YOUR CODE STARTS/ENDS Sections - What to Implement**

#### **1. Feature Transformations**
```python
# Linear transformations for attention
Q = self.WQ(h)
K = self.WK(h)  
V = self.WV(h)

# Reshape for multi-head
g.ndata['Q'] = Q.view(-1, num_heads, head_dim)
g.ndata['K'] = K.view(-1, num_heads, head_dim)
g.ndata['V'] = V.view(-1, num_heads, head_dim)
```

#### **2. Message Passing Coordination**
```python
g.update_all(self.message_func, self.reduce_func)
```

#### **3. Edge Feature Handling**
```python
# For edge-aware models
E = self.WE(e)
g.edata['E'] = E.view(-1, num_heads, head_dim)

# Access in message_func
eij = edges.data['E']
```

#### **4. Attention Score Computation**
```python
# Standard dot-product attention
qikj = (qi * kj).sum(dim=2).unsqueeze(2)

# Edge-modulated attention  
qikj = (qi * eij * kj).sum(dim=2).unsqueeze(2)

# Exponentiate for softmax
expij = torch.exp(qikj / torch.sqrt(torch.tensor(head_dim)))
```

#### **5. Attention Aggregation**
```python
# Softmax normalization
numerator = (expij * vj).sum(dim=1)
denominator = expij.sum(dim=1)
h = numerator / denominator
```

### **Architecture Selection Guide**

| Architecture | Use Case | Complexity | Key Features |
|-------------|----------|------------|--------------|
| **GCN** | Simple graphs, interpretability | Low | Degree normalization, uniform aggregation |
| **GAT** | Heterogeneous graphs, attention | Medium | Learned attention weights, multi-head |
| **GatedGCN** | Edge-rich graphs, molecules | High | Edge updates, gating mechanism |
| **Vanilla GT** | Large graphs, scalability | Medium | Self-attention without structure bias |
| **GT + PE** | Structure-aware tasks | Medium-High | Positional encoding, structural awareness |
| **GT + Edges** | Edge-attributed graphs | High | Edge-modulated attention, dual updates |

### **Common Debugging Tips**

1. **Dimension Mismatches**: Always check tensor shapes in message/reduce functions
2. **NaN Values**: Often caused by division by zero in attention normalization
3. **Memory Issues**: Dense Graph Transformers (Code04) are memory-intensive
4. **Sparse vs Dense**: DGL (Code01-03) vs PyTorch dense operations (Code04)
5. **Edge Updates**: Only GatedGCN and GT with edges modify edge features

---

## Performance & Memory Considerations

### **Memory Complexity**
- **GCN**: O(|V| + |E|) - most efficient
- **GAT**: O(|E| × heads) - attention overhead  
- **GatedGCN**: O(|V| + |E|) - dual updates but sparse
- **GT (DGL)**: O(|E| × heads × d) - sparse attention
- **GT (Dense)**: O(|V|² × heads × d) - dense attention (memory intensive!)

### **When to Use What**

**For Exam/Implementation:**
- **Quick prototyping**: Start with GCN
- **Need attention**: Use GAT
- **Molecular graphs**: GatedGCN or GT with edges
- **Large graphs**: Vanilla GT with DGL
- **Small graphs with rich edge features**: Dense GT (Code04)

This guide covers all the essential patterns you'll encounter in both the 08_Graph_Convnets and 10_Graph_Transformers folders, providing a comprehensive reference for both understanding and implementation.