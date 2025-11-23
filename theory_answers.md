# Theory Answers — Assignment 4: Advanced Image Generation

This file contains the written responses for Part 2 (Diffusion Models) and Part 3 (Energy-Based Models).

---

# 2. Theory: Building Blocks of a Diffusion Model

---

## 1. Sinusoidal Time Embedding Formula (Plain Text)

For timestep t and embedding dimension d:

```
emb_i(t) =
    sin( t / 10000^(i/d) )          if i is even
    cos( t / 10000^((i-1)/d) )      if i is odd
```

---

## 2. Embedding for t = 1, dimension = 8

Computed embedding vector:

```
[0.84147, 0.54030, 0.09983, 0.99500,
 0.0099998, 0.99995, 0.00099999, 0.9999995]
```

---

## 3. Relation to Transformer Positional Encodings

**Similarity:**
- Both diffusion models and Transformers use sinusoidal embeddings with multiple frequencies.
- Both provide a structured notion of “position” to the model.

**Difference:**
- Transformers encode token index (sequence position).
- Diffusion models encode timestep (noise level).
- Diffusion embeddings condition every UNet block, while Transformers add positional encodings once.

---

## 4. Downsampling Resolution

Three stride-2 convolutions:

```
64 -> 32 -> 16 -> 8
```

Final resolution: **8 x 8**

---

## 5. UNet Output and Loss

UNet predicts the added noise:

```
epsilon_hat = UNet(x_t, t)
```

Training objective:

```
Loss = MSE( epsilon, epsilon_hat )
```

---

# 3. Theory: Building Blocks of an Energy Model

---

## 6. Basic Gradient Calculations

Given:

```
y = x^2 + 3x
```

### a) Gradient at x = 2:

```
dy/dx = 2x + 3 = 7
```

### b) If requires_grad=False:
- No gradient tracking
- x.grad remains None

### c) If requires_grad is omitted:
Default is False; no gradients tracked.

---

## 7. Gradients with Learnable Weights

Given:

```python
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([1.0, 3.0])
y = w[0] * x**2 + w[1] * x
```

### a) Why is w.grad = None?
Because w does not have requires_grad=True.

### b) Correct version with gradient tracking:

```python
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([1.0, 3.0], requires_grad=True)

y = w[0] * x**2 + w[1] * x
y.backward()

# w.grad = [4, 2]
```

### c) If requires_grad omitted:
Gradients will not be computed.

---

## 8. Breaking the Graph with detach()

```
z = y.detach()
```

This removes z from the graph; gradients cannot flow back.

### Fixed version:

```python
z = y.detach().requires_grad_(True)
```

---

## 9. Gradient Accumulation

Example:

```python
x = torch.tensor([1.0], requires_grad=True)
y1 = x * 2
y1.backward()
y2 = x * 3
y2.backward()
```

Result:

```
x.grad = 2 + 3 = 5
```

### How to avoid accumulation:

```python
x.grad = None
# or
x.grad.zero_()
```

---
